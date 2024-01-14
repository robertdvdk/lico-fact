
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from models.modules.sinkhorn_distance import SinkhornDistance
from utils.misc import *
import argparse


parser = argparse.ArgumentParser(description='Train LICO')

'''
parser.add_argument('--model', type=str, default='ViT-B/32', help='Pre-trained CLIP model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
parser.add_argument('--image_dim', type=int, default=512, help='Dimension of the image encoder')
parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the classification head')
parser.add_argument('--save_path', type=str, default='fine_tuned_clip.pt', help='Path to save the fine-tuned model')
'''

parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--alpha', type=float, default=10, help='Weight of manifold loss')
parser.add_argument('--beta', type=float, default=1, help='Weight of OT loss')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--sinkhorn_eps', type=float, default=0.1, help='Default eps to use for sinkhorn algorithm')
parser.add_argument('--sinkhorn_max_iters', type=int, default=1000, help='Max iterations for sinkhorn algorithm')
parser.add_argument('--train_dataset', type=str, default='CIFAR10', help='Which dataset to train on')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wrn_builder = build_WideResNet(1, 10, 2, 0.01, 0.1, 0.5)
wrn = wrn_builder.build(10)
wrn = wrn.to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


if args.train_dataset == 'CIFAR10':

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
else:

    print("Invalid training dataset chosen, defaulting to CIFAR10")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training loop

num_epochs = args.num_epochs
w_distance = SinkhornDistance(args.sinkhorn_eps, args.sinkhorn_max_iters)

CELoss = nn.CrossEntropyLoss()
optimizer = SGD(wrn.parameters(), lr=args.lr, momentum=args.momentum)
alpha = args.alpha
beta = args.beta

total_loss = 0

for epoch in range(num_epochs):

    for batch in trainloader:

        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # out: output logits, emb_matrix: similarity matrix of the feature maps
        # emb: unrolled feature maps, w_loss: OT loss
        # label_distribution: similarity matrix of the embedded prompts

        out, emb_matrix, emb, w_loss, label_distribution = wrn(x, targets=y, w_distance=w_distance)

        # cross-entropy loss
        ce_loss = CELoss(out, y)

        # manifold loss
        m_loss = calculate_manifold_loss(label_distribution, emb_matrix)

        # get the full loss

        loss = ce_loss + alpha*m_loss + beta*w_loss

        # training step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {round(total_loss, 3)}")

    test_acc = test_accuracy(wrn, testloader, device)
    
    print(f"Epoch {epoch+1} test accuracy: {round(test_acc, 3)}")

    total_loss = 0
        
