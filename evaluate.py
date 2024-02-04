import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from utils.misc import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate on test set')

    parser.add_argument('--test_dataset', type=str, default='cifar10', help='Which dataset to train on')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model')
    parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--depth', type=int, default=28, help='WRN depth')
    parser.add_argument('--width', type=int, default=2, help='WRN width')
    parser.add_argument('--fixed_temperature', default=False, action='store_true',
                        help="Whether to use a fixed softmax temperature")
    parser.add_argument('--data_root', type=str, default='../../data/', help='Path to data')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare dataset
    dataset_statistics = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'imagenette_160': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'imagenette_320': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }

    assert args.test_dataset in dataset_statistics.keys(), print('Invalid dataset name')

    if args.test_dataset == 'cifar10':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_statistics[args.train_dataset][0],
                                 dataset_statistics[args.train_dataset][1])
        ])
        testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True,
                                               transform=test_transform)
        classes = sorted(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    elif args.test_dataset == 'cifar100':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_statistics[args.train_dataset][0],
                                 dataset_statistics[args.train_dataset][1])
        ])
        testset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True,
                                                transform=test_transform)
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

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    wrn_builder = build_WideResNet(args.depth, args.width, 0.5, args.fixed_temperature)
    wrn = wrn_builder.build(classes)
    model = wrn.to(device)

    # load the model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    acc = get_accuracy(model, testloader, device)
    print(f"Test accuracy: {round(acc, 3)}")

if __name__ == '__main__':
    main()