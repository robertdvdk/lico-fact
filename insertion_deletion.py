'''
Calculate the insertion/deletion score on a given dataset for a trained model
'''

import torch
import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from utils.misc import *
import argparse
import sys
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils.RISE import *
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from utils.misc import *
from utils.GradCAM import apply_grad_cam
from utils.GradCAM_pp import apply_grad_cam_pp

parser = argparse.ArgumentParser(description='Insertion deletion values')

parser.add_argument('--test_dataset', type=str, default='CIFAR10', help='Which dataset to train on')
parser.add_argument('--model_path', type=str, default=None, help='Path to model')
parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
parser.add_argument('--pixel_batch_size', type=int, default=10, help='Number of pixels to insert/delete at a time')
parser.add_argument('--sigma', type=float, default=5., help='Sigma of GaussianBlur')
parser.add_argument('--saliency_method', type=str, default='RISE', help='Which method to use to obtain saliency maps')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build model (ResNet)
wrn_builder = build_WideResNet(1, 10, 2, 0.01, 0.1, 0.5)
model = wrn_builder.build(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
model = model.to(device)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prepare dataset
if args.test_dataset == 'CIFAR10':
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    print("Invalid training dataset chosen")
    sys.exit()

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
# load the model
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
print("Successfully loaded model")


def get_saliency_maps(images, method='random', generator=None, targets=None, target_layer=None):
    if method == 'random':
        # just do random saliency maps for now to test
        heatmap = torch.rand((images.shape[0], images.shape[2], images.shape[3]))
    elif method == 'RISE':
        # for the implementation we use we need to do RISE individually for each image
        heatmap = torch.empty((images.shape[0], images.shape[2], images.shape[3]))
        for i in range(images.shape[0]):
            x = images[i, ...]
            y = targets[i]
            cur_saliency_map = generator.forward(x)[int(y), :, :]
            heatmap[i, ...] = cur_saliency_map.unsqueeze(0)
    elif method == 'GradCAM':
        heatmap = apply_grad_cam(model, target_layer, images)
    elif method == 'GradCAM++':
        heatmap = apply_grad_cam_pp(model, target_layer, images)
    else:
        heatmap = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))
    return heatmap


def insertion(images, saliency_maps, indices, targets, model, pixel_batch_size, blur):
    '''
    images: (B, C, H, W)
    saliency maps: (B, C, H, W)
    indices: (B, HW, 2) [(b, k, :) gives the location of the kth highest saliency in image b]
    targets: (B, )
    pixel_batch_size: int of how many pixels to insert at each step
    blur: GaussianBlur

    returns a (B, K) tensor of insertion values for each image
    '''

    B, C, H, W = images.shape

    inputs = blur(images)

    num_pixels = H*W
    num_inserts = int(num_pixels/pixel_batch_size) + 1 # rounding up

    replaced_pixels = 0

    probs = torch.zeros((B, num_inserts))

    for i in range(num_inserts):

        # get class probabilities

        with torch.no_grad():
            logits = model.get_logits(inputs)
            cur_probs = torch.gather(torch.softmax(logits, dim=-1), 1, targets.unsqueeze(1))
            probs[:, i] = cur_probs.squeeze(1)

            # how many pixels to insert
            cur_batch_size = min(num_pixels - replaced_pixels, pixel_batch_size)

            cur_indices = indices[:, replaced_pixels:replaced_pixels + cur_batch_size, :]

            batch_indices = torch.arange(B)[:, None]

            pixels = images[batch_indices, :, cur_indices[..., 0], cur_indices[..., 1]]

            inputs[batch_indices, :, cur_indices[..., 0], cur_indices[..., 1]] = pixels

            replaced_pixels += cur_batch_size
    
    return probs


def deletion(images, saliency_maps, indices, targets, model, pixel_batch_size):
    '''
    images: (B, C, H, W)
    saliency maps: (B, C, H, W)
    indices: (B, HW, 2) [(b, k, :) gives the location of the kth highest saliency in image b]
    targets: (B, )
    pixel_batch_size: int of how many pixels to delete at each step

    returns a (B, K) tensor of insertion values for each image
    '''

    B, C, H, W = images.shape

    inputs = images

    num_pixels = H*W
    num_deletions = int(num_pixels/pixel_batch_size) + 1 # rounding up

    replaced_pixels = 0

    probs = torch.zeros((B, num_deletions))

    for i in range(num_deletions):
        
        with torch.no_grad():
            # get class probabilities
            logits = model.get_logits(inputs)
            cur_probs = torch.gather(torch.softmax(logits, dim=-1), 1, targets.unsqueeze(1))
            probs[:, i] = cur_probs.squeeze(1)

            # how many pixels to delete
            cur_batch_size = min(num_pixels - replaced_pixels, pixel_batch_size)

            cur_indices = indices[:, replaced_pixels:replaced_pixels + cur_batch_size, :]

            batch_indices = torch.arange(B)[:, None]

            inputs[batch_indices, :, cur_indices[..., 0], cur_indices[..., 1]] = 0

            replaced_pixels += cur_batch_size
    
    return probs


def viz_insertion_deletion(probs, plot_type='Unspecified', filename='plot.png'):
    x = np.linspace(0, 1, len(probs))
    plt.clf()
    plt.scatter(x, probs)
    plt.plot(x, probs)

    plt.title(f'{plot_type} plot')

    plt.savefig(filename)
        

# test out RISE for a single example

'''

rise = RISE(model, input_size=(32,32), initial_mask_size=(7,7))

x, y = next(iter(testloader))
x, y = x.to(device), y.to(device)

x = x[1, ...].unsqueeze(0)
y = y[1].unsqueeze(0)

saliency_maps = rise.forward(x)[int(y), :, :]
print(saliency_maps.shape)
H, W = saliency_maps.shape
num_pixels = H*W

flat_saliency_maps = saliency_maps.view(1, -1)

_, indices = torch.topk(flat_saliency_maps, num_pixels, dim=1)
indices = torch.stack((indices // W, indices % W), dim=-1)

# get insertion and deletion probs

insertion_probs = insertion(x, saliency_maps, indices, y, model, 10, blur)
deletion_probs = deletion(x, saliency_maps, indices, y, model, 10)

insertion_probs = insertion_probs.squeeze().detach().numpy()
deletion_probs = deletion_probs.squeeze().detach().numpy()

viz_insertion_deletion(insertion_probs, "Insertion", './plots/RISE_insertion_test.png')
viz_insertion_deletion(deletion_probs, "Deletion", './plots/RISE_deletion_test.png')

'''


if args.saliency_method == 'RISE':
    generator = RISE(model, input_size=(32, 32), initial_mask_size=(7, 7))
else:
    generator = None

blur = GaussianBlur(int(2 * args.sigma - 1), args.sigma)

num_batches = 0
running_avg_auc_insertion = 0
running_avg_auc_deletion = 0

for batch in testloader:
    x, y = batch
    x, y = x.to(device), y.to(device)

    # get necessary prereqs for insertion and deletion scores
    target_layer = model.block3.layer[0].conv2

    saliency_maps = get_saliency_maps(x, generator=generator, method=args.saliency_method, targets=y, target_layer=target_layer)

    # Convert the tensor to PIL Image and save
    to_pil = transforms.ToPILImage()
    img = to_pil(saliency_maps[0])
    img.save('./outputs/{}_map_{}.png'.format(args.saliency_method, num_batches))
    img = to_pil(x[0])
    img.save('./outputs/{}_img_{}.png'.format(args.saliency_method, num_batches))

    B, H, W = saliency_maps.shape
    num_pixels = H*W

    flat_saliency_maps = saliency_maps.view(B, -1)

    _, indices = torch.topk(flat_saliency_maps, num_pixels, dim=1)
    indices = torch.stack((indices // W, indices % W), dim=-1)

    # get insertion and deletion probs
    insertion_probs = insertion(x, saliency_maps, indices, y, model, 10, blur)
    deletion_probs = deletion(x, saliency_maps, indices, y, model, 10)
    
    # get the auc of insertion probs
    dx = 1/insertion_probs.shape[1]
    auc_insertion = torch.sum((insertion_probs[:, 1:] + insertion_probs[:, :-1]) * dx / 2, axis=1)
    cur_avg_auc_insertion = torch.mean(auc_insertion)

    # get the auc of deletion
    dx = 1/deletion_probs.shape[1]
    auc_deletion = torch.sum((deletion_probs[:, 1:] + deletion_probs[:, :-1]) * dx / 2, axis=1)
    cur_avg_auc_deletion = torch.mean(auc_deletion)

    running_avg_auc_insertion += cur_avg_auc_insertion
    running_avg_auc_deletion += cur_avg_auc_deletion

    num_batches += 1

    torch.cuda.empty_cache()

    #if num_batches == 1:
    #   break


avg_auc_insertion = running_avg_auc_insertion/num_batches
avg_auc_deletion = running_avg_auc_deletion/num_batches

print(f"Average AUC insertion for given model/dataset: {avg_auc_insertion}")
print(f"Average AUC deletion for given model/dataset: {avg_auc_deletion}")
print(f"Average AUC insertion - deletion for given model/dataset: {avg_auc_insertion - avg_auc_deletion}")

