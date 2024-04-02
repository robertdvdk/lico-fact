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
from utils.data import ImagenetteDataset, PartImageNetClassificationDataset
from torch.utils.tensorboard import SummaryWriter
import json


def train(net, trainloader, valloader, optimizer, scheduler, alpha, beta, w_distance, num_epochs, device, writer,
          full_model_save_path, save_model_name):
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    CELoss = nn.CrossEntropyLoss()

    best_model = None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        num_batches = 0
        running_loss = 0
        running_m_loss = 0
        running_OT_loss = 0
        running_CE_loss = 0

        net.train()
        for batch in tqdm(trainloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # out: output logits, emb_matrix: similarity matrix of the feature maps
            # emb: unrolled feature maps, w_loss: OT loss
            # label_distribution: similarity matrix of the embedded prompts
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                out, AF, w_loss, AG = net(x, targets=y, w_distance=w_distance)

                # cross-entropy loss
                ce_loss = CELoss(out, y)

                # manifold loss
                m_loss = calculate_manifold_loss(AG, AF)

                # get the full loss
                loss = ce_loss + alpha * m_loss + beta * w_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            running_loss += loss.item()
            running_m_loss += alpha * float(m_loss)
            running_OT_loss += beta * float(w_loss)
            running_CE_loss += float(ce_loss)
            num_batches += 1

        net.eval()
        avg_loss = running_loss / num_batches

        avg_m_loss = running_m_loss / num_batches
        avg_OT_loss = running_OT_loss / num_batches
        avg_CE_loss = running_CE_loss / num_batches

        writer.add_scalar('Average loss', round(avg_loss, 3), epoch)
        writer.add_scalar('Average manifold loss', round(avg_m_loss, 3), epoch)
        writer.add_scalar('Average OT loss', round(avg_OT_loss, 3), epoch)
        writer.add_scalar('Average CE loss', round(avg_CE_loss, 3), epoch)

        test_acc = get_accuracy(net, valloader, device)
        writer.add_scalar('Validation accuracy', round(test_acc, 3), epoch)

        # calculate val loss to do model selection

        val_loss = get_loss(net, valloader, device, w_distance, alpha, beta)

        if val_loss < best_val_loss:
            best_model = net.state_dict()
            best_val_loss = val_loss

    if not os.path.exists(os.path.dirname(full_model_save_path)):
        os.makedirs(os.path.dirname(full_model_save_path))

    torch.save(best_model, f'{full_model_save_path}/{save_model_name}.pt')

    writer.flush()


def main():
    parser = argparse.ArgumentParser(description='Train LICO')

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
    parser.add_argument('--fixed_temperature', default=False, action='store_true',
                        help="Whether to use a fixed softmax temperature")
    parser.add_argument('--image_feature_dim', default=64, type=int,
                        help="Dimension of feature maps")

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

    dataset_statistics = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'imagenette_160': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'imagenette_320': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'partimagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }

    dataset_image_sizes = {
        'cifar10': 32,
        'cifar100': 32,
        'imagenette_320': 320,
        'imagenette_160': 160,
        'partimagenet': 224,
        'imagenet': 224
    }

    assert args.train_dataset in dataset_statistics.keys(), print('Invalid dataset name')

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(dataset_image_sizes[args.train_dataset], padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(dataset_statistics[args.train_dataset][0],
                             dataset_statistics[args.train_dataset][1])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_statistics[args.train_dataset][0],
                             dataset_statistics[args.train_dataset][1])
    ])

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

        classnames = sorted(['plane', 'car', 'bird', 'cat',
                      'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

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

        classnames = sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',
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
    elif args.train_dataset == 'partimagenet':
        # Different transforms for ImageNet
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.RandomCrop(dataset_image_sizes[args.train_dataset]),
            transforms.Normalize(dataset_statistics[args.train_dataset][0],
                                 dataset_statistics[args.train_dataset][1])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(dataset_image_sizes[args.train_dataset]),
            transforms.Normalize(dataset_statistics[args.train_dataset][0],
                                 dataset_statistics[args.train_dataset][1])
        ])
        trainset = PartImageNetClassificationDataset(root=args.data_root + 'partimagenet/',
                                                     split='train', transform=train_transform)
        valset = PartImageNetClassificationDataset(root=args.data_root + 'partimagenet/',
                                                   split='val', transform=val_transform)
        testset = PartImageNetClassificationDataset(root=args.data_root + 'partimagenet/',
                                                    split='test', transform=val_transform)

        classnames = sorted(['cairn', 'patas', 'anemone fish', 'barracouta', 'tractor', 'howler monkey', 'beach wagon', 'otter', 'Gila monster', 'jacamar', 'box turtle', 'hognose snake', 'Brittany spaniel', 'alligator lizard', 'bighorn', 'schooner', 'squirrel monkey', 'kite', 'cheetah', 'yawl', 'puffer', 'vine snake', 'coucal', 'marmoset', 'bicycle-built-for-two', 'ibex', 'badger', 'diamondback', 'American black bear', 'Arabian camel', 'frilled lizard', 'Weimaraner', 'moped', 'weasel', 'orangutan', 'trimaran', 'limousine', 'macaque', 'mink', 'bee eater', 'unicycle', 'gorilla', 'proboscis monkey', 'snowplow', 'tree frog', 'loggerhead', 'boa constrictor', 'Irish water spaniel', 'capuchin', 'garter snake', 'golfcart', 'recreational vehicle', 'African crocodile', 'gibbon', 'convertible', 'mud turtle', 'Walker hound', 'terrapin', 'green lizard', 'night snake', 'colobus', 'ringneck snake', 'brown bear', 'goldfish', 'polecat', 'tricycle', 'common newt', 'Tibetan terrier', 'pirate', 'tench', 'minivan', 'Boston bull', 'cougar', 'warplane', 'great white shark', 'golden retriever', 'warthog', 'airliner', 'giant panda', 'green mamba', 'sloth bear', 'ice bear', 'sidewinder', 'little blue heron', 'American egret', 'redbone', 'sports car', 'tailed frog', 'African chameleon', 'Indian cobra', 'titi', 'English springer', 'siamang', 'Saint Bernard', 'jeep', 'horned viper', 'albatross', 'dowitcher', 'spoonbill', 'bald eagle', 'chimpanzee', 'ruddy turnstone', 'coho', 'police van', 'timber wolf', 'hartebeest', 'ambulance', 'water bottle', 'rock python', 'leopard', 'American alligator', 'beer bottle', 'Komodo dragon', 'ox', 'racer', 'Saluki', 'whiptail', 'wine bottle', 'vizsla', 'tiger', 'agama', 'baboon', 'European gallinule', 'chow', 'spotted salamander', 'king snake', 'mountain bike', 'Japanese spaniel', 'cab', 'black stork', 'ram', 'garbage truck', 'hammerhead', 'green snake', 'Arctic fox', 'tiger shark', 'guenon', 'go-kart', 'Egyptian cat', 'minibus', 'pill bottle', 'impala', 'soft-coated wheaten terrier', 'fox squirrel', 'thunder snake', 'spider monkey', 'killer whale', 'water buffalo', 'goose', 'Eskimo dog', 'leatherback turtle', 'Gordon setter', 'pop bottle', 'bullfrog', 'gazelle', 'trolleybus', 'school bus', 'motor scooter'], key = lambda x: x.lower())

    else:
        image_size = int(args.train_dataset.split("_")[-1])
        trainset_full = ImagenetteDataset(args.data_root + args.train_dataset, image_size,
                                          download=True, validation=False, transform=train_transform)

        num_train = len(trainset_full)
        num_val = int(num_train * args.val_prop)
        num_train = num_train - num_val

        trainset, valset = torch.utils.data.random_split(trainset_full, [num_train, num_val])

        valset.transforms = val_transform

        testset = ImagenetteDataset(args.data_root + args.train_dataset, image_size,
                                    download=True, validation=True, transform=val_transform)

        classnames = sorted(["tench", "English springer", "cassette player", "chain saw",
                      "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"])


    if args.train_dataset == 'partimagenet':
        net = ResNetPrompt(classnames, BasicBlock, [2, 2, 2, 2])
        net = net.to(device)
    else:
        wrn_builder = build_WideResNet(args.depth, args.width, 0.5, fixed_temperature=args.fixed_temperature, image_feature_dim=args.image_feature_dim)
        net = wrn_builder.build(classnames)
        net = net.to(device)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    num_epochs = args.num_epochs
    num_steps = num_epochs * int(len(trainset) / args.batch_size)

    w_distance = SinkhornDistance(args.sinkhorn_eps, args.sinkhorn_max_iters)

    # Freeze the clip model
    for name, param in net.named_parameters():
        if name.startswith('clip'):
            param.requires_grad = False

    optimizer = SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = CosineAnnealingStepLR(optimizer, T_max=num_steps)

    train(net, trainloader, valloader, optimizer, scheduler, args.alpha, args.beta, w_distance, num_epochs, device,
          writer, full_model_save_path, args.save_model_name)


if __name__ == "__main__":
    main()