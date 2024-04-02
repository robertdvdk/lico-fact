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

def evaluate(net, testloader, maskloader, device):
    print("Accuracy:", get_accuracy(net, testloader, device))
    ious = []
    for img, label, mask in maskloader:
        saliency_map = get_saliency_maps(net, img, label, target_layers=[net.layer4[1].conv2])
        iou = torch.sum(saliency_map * mask) / torch.sum(saliency_map + mask - saliency_map * mask)
        ious.append(iou)
    print(np.mean(ious))

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
        transforms.CenterCrop(224)
    ])

    testset = PartImageNetClassificationDataset('../../data/partimagenet/',
                                                transform=test_transform,
                                                split='test',
                                                return_masks=False)
    maskset = PartImageNetClassificationDataset('../../data/partimagenet/',
                                                transform=test_transform,
                                                split='test',
                                                return_masks=True)

    classnames = ['cairn', 'patas', 'anemone fish', 'barracouta', 'tractor', 'howler monkey', 'beach wagon', 'otter',
                  'Gila monster', 'jacamar', 'box turtle', 'hognose snake', 'Brittany spaniel', 'alligator lizard',
                  'bighorn', 'schooner', 'squirrel monkey', 'kite', 'cheetah', 'yawl', 'puffer', 'vine snake', 'coucal',
                  'marmoset', 'bicycle-built-for-two', 'ibex', 'badger', 'diamondback', 'American black bear',
                  'Arabian camel', 'frilled lizard', 'Weimaraner', 'moped', 'weasel', 'orangutan', 'trimaran',
                  'limousine', 'macaque', 'mink', 'bee eater', 'unicycle', 'gorilla', 'proboscis monkey', 'snowplow',
                  'tree frog', 'loggerhead', 'boa constrictor', 'Irish water spaniel', 'capuchin', 'garter snake',
                  'golfcart', 'recreational vehicle', 'African crocodile', 'gibbon', 'convertible', 'mud turtle',
                  'Walker hound', 'terrapin', 'green lizard', 'night snake', 'colobus', 'ringneck snake', 'brown bear',
                  'goldfish', 'polecat', 'tricycle', 'common newt', 'Tibetan terrier', 'pirate', 'tench', 'minivan',
                  'Boston bull', 'cougar', 'warplane', 'great white shark', 'golden retriever', 'warthog', 'airliner',
                  'giant panda', 'green mamba', 'sloth bear', 'ice bear', 'sidewinder', 'little blue heron',
                  'American egret', 'redbone', 'sports car', 'tailed frog', 'African chameleon', 'Indian cobra', 'titi',
                  'English springer', 'siamang', 'Saint Bernard', 'jeep', 'horned viper', 'albatross', 'dowitcher',
                  'spoonbill', 'bald eagle', 'chimpanzee', 'ruddy turnstone', 'coho', 'police van', 'timber wolf',
                  'hartebeest', 'ambulance', 'water bottle', 'rock python', 'leopard', 'American alligator',
                  'beer bottle', 'Komodo dragon', 'ox', 'racer', 'Saluki', 'whiptail', 'wine bottle', 'vizsla', 'tiger',
                  'agama', 'baboon', 'European gallinule', 'chow', 'spotted salamander', 'king snake', 'mountain bike',
                  'Japanese spaniel', 'cab', 'black stork', 'ram', 'garbage truck', 'hammerhead', 'green snake',
                  'Arctic fox', 'tiger shark', 'guenon', 'go-kart', 'Egyptian cat', 'minibus', 'pill bottle', 'impala',
                  'soft-coated wheaten terrier', 'fox squirrel', 'thunder snake', 'spider monkey', 'killer whale',
                  'water buffalo', 'goose', 'Eskimo dog', 'leatherback turtle', 'Gordon setter', 'pop bottle',
                  'bullfrog', 'gazelle', 'trolleybus', 'school bus', 'motor scooter']

    net = ResNetPrompt(classnames, BasicBlock, [2, 2, 2, 2], mode='test')
    net.load_state_dict(torch.load('./trained_models/pim_baseline_run1/pim_baseline_run1_199.pt'))
    net = net.to(device)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    maskloader = torch.utils.data.DataLoader(maskset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    evaluate(net, testloader, maskloader, device)


if __name__ == "__main__":
    main()