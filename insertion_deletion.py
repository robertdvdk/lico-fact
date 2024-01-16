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

parser = argparse.ArgumentParser(description='Train LICO')

parser.add_argument('--test_dataset', type=str, default='CIFAR10', help='Which dataset to train on')
parser.add_argument('--model_path', type=str, default=None, help='Path to model')
parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
parser.add_argument('--pixel_batch_size', type=int, default=10, help='Number of pixels to insert/delete at a time')
parser.add_argument('--sigma', type=float, default=5., help='Sigma of GaussianBlur')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wrn_builder = build_WideResNet(1, 10, 2, 0.01, 0.1, 0.5)
model = wrn_builder.build(10)
model = model.to(device)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if args.test_dataset == 'CIFAR10':

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    
    
            
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
else:

    print("Invalid training dataset chosen")
    sys.exit()

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)
    
# load the model

model.load_state_dict(torch.load(args.model_path, map_location=device))

model.eval()

print("Successfully loaded model")

def get_saliency_maps(images):
    ''' just do random saliency maps for now to test'''
    return torch.rand((images.shape[0], images.shape[2], images.shape[3]))

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
        


blur = GaussianBlur(int(2 * args.sigma - 1), args.sigma)

num_batches = 0
running_avg_auc_insertion = 0
running_avg_auc_deletion = 0

for batch in testloader:

    x, y = batch
    x, y = x.to(device), y.to(device)

    # get necessary prereqs for insertion and deletion scores
    saliency_maps = get_saliency_maps(x)

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

avg_auc_insertion = running_avg_auc_insertion/num_batches
avg_auc_deletion = running_avg_auc_deletion/num_batches

print(f"Average AUC insertion for given model/dataset: {avg_auc_insertion}")
print(f"Average AUC deletion for given model/dataset: {avg_auc_deletion}")
print(f"Average AUC insertion - deletion for given model/dataset: {avg_auc_insertion - avg_auc_deletion}")






    



    







    

