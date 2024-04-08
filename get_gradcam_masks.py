import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from models.resnet_prompt import *
from utils.misc import *
import argparse
from torch.utils.tensorboard import SummaryWriter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import os

def get_maps(net, testloader, model_name):
    if not os.path.exists(f'./results/{model_name}'):
        os.mkdir(f'./results/{model_name}')

    i = 0
    for img, label in testloader:
        plt.imshow((img.squeeze().permute(1, 2, 0) * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])))
        plt.savefig(f'./results/{model_name}/input_unnormalized_{i}.png')
        saliency_map = get_saliency_maps(net, img, label, target_layers=[net.layer4[2].conv1])
        plt.imshow((img.squeeze().permute(1, 2, 0) * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])))
        plt.imshow(saliency_map.squeeze(), cmap='jet', alpha=0.5)
        plt.savefig(f'./results/{model_name}/saliency_map_{i}.png')
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

    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_root', type=str, default='../../data/', help='Path to data')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')

    args = parser.parse_args()

    # Ensure the training is reproducible
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    get_maps(net, testloader, 'IMNET50')


if __name__ == "__main__":
    main()