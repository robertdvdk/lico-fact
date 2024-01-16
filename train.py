
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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

parser = argparse.ArgumentParser(description='Train LICO')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(25)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():

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
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=10, help='Weight of manifold loss')
    parser.add_argument('--beta', type=float, default=1, help='Weight of OT loss')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--val_prop', type=float, default=0.05, help='Proportion of train data to use for validation')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.1, help='Default eps to use for sinkhorn algorithm')
    parser.add_argument('--sinkhorn_max_iters', type=int, default=1000, help='Max iterations for sinkhorn algorithm')
    parser.add_argument('--train_dataset', type=str, default='cifar10', help='Which dataset to train on')
    parser.add_argument('--save_path', type=str, default='./trained_models/', help='Path to save trained models')
    parser.add_argument('--save_model_name', type=str, default='wrn28-2.pt', help='Model name to save')
    parser.add_argument('--depth', type=int, default=28, help='WideResNet depth')
    parser.add_argument('--width', type=int, default=2, help='WideResNet widening factor')
    parser.add_argument('--data_root', type=str, default='../../data/', help='Path to data')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')

    args = parser.parse_args()

    writer = SummaryWriter(log_dir=f'{args.save_path}/{args.save_model_name}')
    writer.add_text('Dataset', args.train_dataset)
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Momentum', str(args.momentum))
    writer.add_text('Weight decay', str(args.weight_decay))
    writer.add_text('Number of epochs', str(args.num_epochs))
    writer.add_text('Alpha', str(args.alpha))
    writer.add_text('Beta', str(args.beta))
    writer.add_text('Batch size', str(args.batch_size))



    with open(f'{args.save_path}/{args.save_model_name}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_statistics = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    }

    try:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(dataset_statistics[args.train_dataset][0],
                                 dataset_statistics[args.train_dataset][1])
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_statistics[args.train_dataset][0],
                                 dataset_statistics[args.train_dataset][1])
        ])

    except KeyError:
        print("Invalid training dataset chosen.")
        return

    if args.train_dataset == 'cifar10':
        trainset_full = torchvision.datasets.CIFAR10(root=args.data_root + args.train_dataset, train=True,
                                                     download=True, transform=train_transform)

        num_train = len(trainset_full)
        num_val = int(num_train * args.val_prop)
        num_train = num_train - num_val

        trainset, valset = torch.utils.data.random_split(trainset_full, [num_train, num_val])

        valset.transforms = val_transform

        testset = torchvision.datasets.CIFAR10(root=args.data_root + args.train_dataset, train=False,
                                               download=True, transform=val_transform)



        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        num_classes = 10

    elif args.train_dataset == 'cifar100':
        trainset_full = torchvision.datasets.CIFAR100(root=args.data_root + args.train_dataset, train=True,
                                                     download=True, transform=train_transform)

        num_train = len(trainset_full)
        num_val = int(num_train * args.val_prop)
        num_train = num_train - num_val

        trainset, valset = torch.utils.data.random_split(trainset_full, [num_train, num_val])

        valset.transforms = val_transform

        testset = torchvision.datasets.CIFAR100(root=args.data_root + args.train_dataset, train=False,
                                               download=True, transform=val_transform)

        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        num_classes = 100

    wrn_builder = build_WideResNet(1, args.depth, args.width, 0.01, 0.1, 0.5)
    wrn = wrn_builder.build(num_classes)
    wrn = wrn.to(device)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    valloader= torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)


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
        for batch in tqdm(trainloader):

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



        wrn.eval()
        avg_loss = running_loss/num_batches

        avg_m_loss = running_m_loss/num_batches
        avg_OT_loss = running_OT_loss/num_batches
        avg_CE_loss = running_CE_loss/num_batches

        writer.add_scalar('Average loss', round(avg_loss, 3), epoch)
        writer.add_scalar('Average manifold loss', round(avg_m_loss, 3), epoch)
        writer.add_scalar('Average OT loss', round(avg_OT_loss, 3), epoch)
        writer.add_scalar('Average CE loss', round(avg_CE_loss, 3), epoch)

        test_acc = get_accuracy(wrn, valloader, device)
        writer.add_scalar('Validation accuracy', round(test_acc, 3), epoch)

        # calculate val loss to do model selection

        val_loss = get_loss(wrn, valloader, args, device)

        if val_loss < best_val_loss:
            best_model = wrn.state_dict()
            best_val_loss = val_loss


    full_model_save_path = args.save_path + args.save_model_name

    if not os.path.exists(os.path.dirname(full_model_save_path)):
        os.makedirs(os.path.dirname(full_model_save_path))

    torch.save(best_model, full_model_save_path)

    writer.flush()

        
if __name__=="__main__":
    main()