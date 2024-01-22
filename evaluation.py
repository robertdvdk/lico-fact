
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

parser = argparse.ArgumentParser(description='Evaluate on test set')

parser.add_argument('--test_dataset', type=str, default='CIFAR10', help='Which dataset to train on')
parser.add_argument('--model_path', type=str, default=None, help='Path to model')
parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
                                         shuffle=True, num_workers=2)

wrn_builder = build_WideResNet(1, 28, 2, 0.01, 0.1, 0.5)
model = wrn_builder.build(classes)
model = model.to(device)
    
# load the model

model.load_state_dict(torch.load(args.model_path, map_location=device))

model.eval()

acc = get_accuracy(model, testloader, device)

print(f"Test accuracy: {round(acc, 3)}")

