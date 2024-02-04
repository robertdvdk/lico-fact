'''
Calculate the insertion/deletion score on a given dataset for a trained model
'''

import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
import argparse
import sys
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
from utils.RISE import *
from utils.misc import *
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

parser = argparse.ArgumentParser(description='Insertion deletion values')

parser.add_argument('--test_dataset', type=str, default='CIFAR10', help='Which dataset to train on')
parser.add_argument('--model_path', type=str, default=None, help='Path to model')
parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
parser.add_argument('--pixel_batch_size', type=int, default=10, help='Number of pixels to insert/delete at a time')
parser.add_argument('--sigma', type=float, default=5., help='Sigma of GaussianBlur')
parser.add_argument('--saliency_method', type=str, default='RISE', help='Which method to use to obtain saliency maps')
parser.add_argument('--depth', type=int, default=28, help='WRN depth')
parser.add_argument('--width', type=int, default=2, help='WRN width')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Prepare dataset
if args.test_dataset == 'CIFAR10':

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif args.test_dataset == 'CIFAR100':

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    classes = sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',
                      'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                      'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                      'bottles', 'bowls', 'cans', 'cups', 'plates',
                      'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                      'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                      'bed', 'chair', 'couch', 'table', 'wardrobe',
                      'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                      'bear', 'leopard', 'lion', 'tiger', 'wolf',
                      'bridge', 'castle', 'house', 'road', 'skyscraper',
                      'cloud', 'forest', 'mountain', 'plain', 'sea',
                      'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                      'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                      'crab', 'lobster', 'snail', 'spider', 'worm',
                      'baby', 'boy', 'girl', 'man', 'woman',
                      'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                      'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                      'maple', 'oak', 'palm', 'pine', 'willow',
                      'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                      'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'])
else:
    print("Invalid training dataset chosen")
    sys.exit()

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

wrn_builder = build_WideResNet(args.depth, args.width, 0.5)
wrn = wrn_builder.build(classes)
model = wrn.to(device)
    
# load the model

model.load_state_dict(torch.load(args.model_path, map_location=device))

model.eval()

print("Successfully loaded model")


def get_saliency_maps(images, method='random', generator=None, targets=None, target_layers=None):
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
    elif method == 'GradCAM' or method == 'GradCAM++' or method == 'ScoreCAM':
        if method == 'GradCAM':
            cam = GradCAM(model=model, target_layers=target_layers)
        elif method == 'GradCAM++':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        else:
            cam = ScoreCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(item) for item in targets]
        grayscale_cam = cam(input_tensor=images, targets=targets)
        heatmap = torch.tensor(grayscale_cam)
    else:
        heatmap = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))
    return heatmap


def insertion(images, indices, targets, model, pixel_batch_size, blur):
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
            logits = model(inputs)
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


def deletion(images, indices, targets, model, pixel_batch_size):
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
            logits = model(inputs)
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

    target_layers = [model.block3.layer[0].conv2]

    saliency_maps = get_saliency_maps(x, generator=generator, method=args.saliency_method, targets=y, target_layers=target_layers)

    # Convert the tensor to PIL Image and save
    #to_pil = transforms.ToPILImage()
    #img = to_pil(saliency_maps[0])
    #img.save('./outputs/{}_map_{}.png'.format(args.saliency_method, num_batches))
    #img = to_pil(x[0])
    #img.save('./outputs/{}_img_{}.png'.format(args.saliency_method, num_batches))

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

    print(num_batches)


avg_auc_insertion = running_avg_auc_insertion/num_batches
avg_auc_deletion = running_avg_auc_deletion/num_batches

print(f"Average AUC insertion for given model/dataset: {avg_auc_insertion}")
print(f"Average AUC deletion for given model/dataset: {avg_auc_deletion}")
print(f"Average AUC insertion - deletion for given model/dataset: {avg_auc_insertion - avg_auc_deletion}")

