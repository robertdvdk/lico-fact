'''
Use t-SNE to visualise the shared latent space of the embedded text and feature maps

Load the model
Pick a class
Get the embedded prompt tokens for that class
Get a bunch of feature maps for that class

complications:
Text features can be embedded in different ways depending on their order - probably shouldn't matter *too* much though?
We can investigate this later

Just pick XXXX...class for now

'''


import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from models.modules.sinkhorn_distance import SinkhornDistance
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.misc import *
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Latent viz')

parser.add_argument('--test_dataset', type=str, default='CIFAR10', help='Which dataset to test on')
parser.add_argument('--model_path', type=str, default=None, help='Path to model')
parser.add_argument('--num_examples', type=str, default=1000, help='Number of examples to visualise')
parser.add_argument('--viz_text', type=bool, default=True, help='Visualise text embeddings or not')
parser.add_argument('--class', type=int, default=0, help='Which class to visualise')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

args = parser.parse_args()

# load the model
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

    print("Invalid dataset chosen")
    sys.exit()

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)
    
# load the model

model.load_state_dict(torch.load(args.model_path, map_location=device))

model.eval()

print("Successfully loaded model")

# get the text features

prompt_learner = model.prompt_learner

text_features = prompt_learner.forward()

print(text_features.shape)
