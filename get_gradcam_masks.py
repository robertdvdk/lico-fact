from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from models.resnet_prompt import *
from models.modules.sinkhorn_distance import SinkhornDistance
from utils.misc import *
import argparse
import os
from tqdm import tqdm
from utils.data import PartImageNetClassificationDataset
from torch.utils.tensorboard import SummaryWriter
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

def get_maps(net, testloader, device):
    i = 0
    for img, label in testloader:
        plt.imshow((img.squeeze().permute(1, 2, 0) * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])))
        plt.savefig(f'input_unnorm_{i}.png')
        saliency_map = get_saliency_maps(net, img, label, target_layers=[net.layer4[2].conv1])
        plt.imshow(saliency_map.squeeze(), cmap='jet')
        plt.savefig(f'saliency_map_{i}.png')
        i += 1
        if i >= 30:
            return

def get_saliency_maps(net, images, targets=None, target_layers=None):
    cam = GradCAM(model=net, target_layers=target_layers)
    targets = [ClassifierOutputTarget(item) for item in targets]
    grayscale_cam = cam(input_tensor=images, targets=targets)
    heatmap = torch.tensor(grayscale_cam)
    return heatmap

def main():
    parser = argparse.ArgumentParser(description='Test LICO')

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
    parser.add_argument('--save_model_name', type=str, default='wrn28-2', help='Model name to save')
    parser.add_argument('--depth', type=int, default=28, help='WideResNet depth')
    parser.add_argument('--width', type=int, default=2, help='WideResNet widening factor')
    parser.add_argument('--data_root', type=str, default='../../data/', help='Path to data')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')

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

    if args.save_path[-1] != '/':
        args.save_path += '/'
    full_model_save_path = args.save_path + args.save_model_name

    # Ensure the training is reproducible
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(f'{full_model_save_path}/{args.save_model_name}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Different transforms for ImageNet
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    train_split = torchvision.datasets.ImageFolder(root="/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/train",
                                                   transform=test_transform)
    train_size = int(len(train_split) * 0.16)
    val_size, test_size = int(len(train_split) * 0.02), int(len(train_split) * 0.02)
    rest = len(train_split) - train_size - val_size - test_size
    _, _, testset, _ = torch.utils.data.random_split(train_split, [train_size, val_size, test_size, rest])
    classnames = tuple([line.strip() for line in open('./utils/imagenet_classnames.txt', 'r')])

    net = ResNetPrompt(classnames, BasicBlock, [3, 4, 6, 3], mode='test')
    net.load_state_dict(torch.load('./trained_models/IMNET50/IMNET50.pt'))
    net = net.to(device)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    get_maps(net, testloader, device)


if __name__ == "__main__":
    main()