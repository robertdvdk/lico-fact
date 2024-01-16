
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

parser = argparse.ArgumentParser(description='Train LICO')

'''
parser.add_argument('--model', type=str, default='ViT-B/32', help='Pre-trained CLIP model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
parser.add_argument('--image_dim', type=int, default=512, help='Dimension of the image encoder')
parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the classification head')
parser.add_argument('--save_path', type=str, default='fine_tuned_clip.pt', help='Path to save the fine-tuned model')
'''

parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay parameter')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--alpha', type=float, default=10, help='Weight of manifold loss')
parser.add_argument('--beta', type=float, default=1, help='Weight of OT loss')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--val_prop', type=float, default=0.05, help='Proportion of train data to use for validation')
parser.add_argument('--sinkhorn_eps', type=float, default=0.1, help='Default eps to use for sinkhorn algorithm')
parser.add_argument('--sinkhorn_max_iters', type=int, default=1000, help='Max iterations for sinkhorn algorithm')
parser.add_argument('--train_dataset', type=str, default='CIFAR10', help='Which dataset to train on')
parser.add_argument('--save_path', type=str, default='./trained_models/', help='Path to save trained models')
parser.add_argument('--save_model_name', type=str, default='default_model_name.pt', help='Model name to save')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wrn_builder = build_WideResNet(1, 28, 2, 0.01, 0.1, 0.5)
wrn = wrn_builder.build(10)
wrn = wrn.to(device)



train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if args.train_dataset == 'CIFAR10':

    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=train_transform)
    
    num_train = len(trainset_full)
    num_val = int(num_train * args.val_prop) 
    num_train = num_train - num_val

    trainset, valset = torch.utils.data.random_split(trainset_full, [num_train, num_val])

    valset.transforms=val_transform

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)
    
    
            
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
else:

    print("Invalid training dataset chosen, defaulting to CIFAR10")

    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=train_transform)
    
    num_train = len(trainset_full)
    num_val = int(num_train * args.val_prop) 
    num_train = num_train - num_val

    trainset, valset = torch.utils.data.random_split(trainset_full, [num_train, num_val])

    valset.transforms=val_transform

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)
    
    
            
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

valloader= torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training loop

num_epochs = args.num_epochs
num_steps = num_epochs*int(len(trainset)/args.batch_size)

w_distance = SinkhornDistance(args.sinkhorn_eps, args.sinkhorn_max_iters)

CELoss = nn.CrossEntropyLoss()
optimizer = SGD(wrn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = CosineAnnealingStepLR(optimizer, T_max=num_steps)

alpha = args.alpha
beta = args.beta

best_model = None
best_val_loss = float("inf")

for epoch in range(num_epochs):
    
    num_batches = 0
    running_loss = 0

    running_m_loss = 0
    running_OT_loss = 0
    running_CE_loss = 0

    wrn.train()

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

        scheduler.step()

        running_loss += loss.item()
        running_m_loss += alpha*float(m_loss)
        running_OT_loss += beta*float(w_loss)
        running_CE_loss += float(ce_loss)

        num_batches += 1

        '''
        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group['lr'])
        '''


    avg_loss = running_loss/num_batches

    avg_m_loss = running_m_loss/num_batches
    avg_OT_loss = running_OT_loss/num_batches
    avg_CE_loss = running_CE_loss/num_batches


    print(f"Epoch {epoch+1} average loss: {round(avg_loss, 3)}")
    print(f"Epoch {epoch+1} average manifold loss: {round(avg_m_loss, 3)}")
    print(f"Epoch {epoch+1} average OT loss: {round(avg_OT_loss, 3)}")
    print(f"Epoch {epoch+1} average CE loss: {round(avg_CE_loss, 3)}")

    test_acc = get_accuracy(wrn, valloader, device)

    print(f"Epoch {epoch+1} validation accuracy: {round(test_acc, 3)}")

    # calculate val loss to do model selection

    val_loss = get_loss(wrn, valloader, args, device)

    if val_loss < best_val_loss:
        best_model = wrn.state_dict()
        best_val_loss = val_loss


full_model_save_path = args.save_path + args.save_model_name

if not os.path.exists(os.path.dirname(full_model_save_path)):
    os.makedirs(os.path.dirname(full_model_save_path))

torch.save(best_model, full_model_save_path)

        
