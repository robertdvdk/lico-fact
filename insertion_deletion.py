'''
Calculate the insertion/deletion score on a given dataset for a trained model
'''

import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
import argparse
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
from utils.RISE import *
from utils.misc import *
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.data import ImagenetteDataset

def get_saliency_maps(images, model, method='random', generator=None, targets=None, target_layers=None):
    if method == 'random':
        heatmap = torch.rand((images.shape[0], images.shape[2], images.shape[3]))
    elif method == 'RISE':
        heatmap = torch.empty((images.shape[0], images.shape[2], images.shape[3]))
        for i in range(images.shape[0]):
            x = images[i, ...]
            y = targets[i]
            cur_saliency_map = generator.forward(x)[int(y), :, :]
            heatmap[i, ...] = cur_saliency_map.unsqueeze(0)
    elif method == 'GradCAM' or method == 'GradCAM++' or method == 'ScoreCAM':
        if method == 'GradCAM':
            cam = GradCAM(model=model, target_layers=target_layers)
        elif method == 'GradCAM++':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        else:
            cam = ScoreCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(item) for item in targets]
        grayscale_cam = cam(input_tensor=images, targets=targets)
        heatmap = torch.tensor(grayscale_cam)
    else:
        heatmap = torch.zeros((images.shape[0], images.shape[2], images.shape[3]))
    return heatmap


def insertion(images, indices, targets, model, pixel_batch_size, blur):
    '''
    images: (B, C, H, W)
    saliency maps: (B, C, H, W)
    indices: (B, HW, 2) [(b, k, :) gives the location of the kth highest saliency in image b]
    targets: (B, )
    pixel_batch_size: int of how many pixels to insert at each step
    blur: GaussianBlur

    returns a (B, K) tensor of insertion values for each image
    '''

    B, C, H, W = images.shape

    inputs = blur(images)

    num_pixels = H * W
    num_inserts = int(num_pixels / pixel_batch_size) + 1  # rounding up

    replaced_pixels = 0

    probs = torch.zeros((B, num_inserts))

    for i in range(num_inserts):
        # get class probabilities

        with torch.no_grad():
            logits = model(inputs)
            cur_probs = torch.gather(torch.softmax(logits, dim=-1), 1, targets.unsqueeze(1))
            probs[:, i] = cur_probs.squeeze(1)

            # how many pixels to insert
            cur_batch_size = min(num_pixels - replaced_pixels, pixel_batch_size)

            cur_indices = indices[:, replaced_pixels:replaced_pixels + cur_batch_size, :]

            batch_indices = torch.arange(B)[:, None]

            pixels = images[batch_indices, :, cur_indices[..., 0], cur_indices[..., 1]]

            inputs[batch_indices, :, cur_indices[..., 0], cur_indices[..., 1]] = pixels

            replaced_pixels += cur_batch_size

    return probs


def deletion(images, indices, targets, model, pixel_batch_size):
    '''
    images: (B, C, H, W)
    saliency maps: (B, C, H, W)
    indices: (B, HW, 2) [(b, k, :) gives the location of the kth highest saliency in image b]
    targets: (B, )
    pixel_batch_size: int of how many pixels to delete at each step

    returns a (B, K) tensor of insertion values for each image
    '''

    B, C, H, W = images.shape

    inputs = images

    num_pixels = H * W
    num_deletions = int(num_pixels / pixel_batch_size) + 1  # rounding up

    replaced_pixels = 0

    probs = torch.zeros((B, num_deletions))

    for i in range(num_deletions):
        with torch.no_grad():
            # get class probabilities
            logits = model(inputs)
            cur_probs = torch.gather(torch.softmax(logits, dim=-1), 1, targets.unsqueeze(1))
            probs[:, i] = cur_probs.squeeze(1)

            # how many pixels to delete
            cur_batch_size = min(num_pixels - replaced_pixels, pixel_batch_size)

            cur_indices = indices[:, replaced_pixels:replaced_pixels + cur_batch_size, :]

            batch_indices = torch.arange(B)[:, None]

            inputs[batch_indices, :, cur_indices[..., 0], cur_indices[..., 1]] = 0

            replaced_pixels += cur_batch_size

    return probs


def viz_insertion_deletion(probs, plot_type='Unspecified', filename='plot.png'):
    x = np.linspace(0, 1, len(probs))
    plt.clf()
    plt.scatter(x, probs)
    plt.plot(x, probs)

    plt.title(f'{plot_type} plot')

    plt.savefig(filename)


def run_evaluation(testloader, device, model, generator, saliency_method, blur, pixel_batch_size):
    num_batches = 0
    running_avg_auc_insertion = 0
    running_avg_auc_deletion = 0

    for batch in testloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        target_layers = [model.block3.layer[0].conv2]

        saliency_maps = get_saliency_maps(x, model, method=saliency_method, generator=generator, targets=y,
                                          target_layers=target_layers)

        B, H, W = saliency_maps.shape
        num_pixels = H * W

        flat_saliency_maps = saliency_maps.view(B, -1)

        _, indices = torch.topk(flat_saliency_maps, num_pixels, dim=1)
        indices = torch.stack((indices // W, indices % W), dim=-1)

        # get insertion and deletion probs
        insertion_probs = insertion(x, indices, y, model, pixel_batch_size, blur)
        deletion_probs = deletion(x, indices, y, model, pixel_batch_size)

        # get the auc of insertion probs
        dx = 1 / insertion_probs.shape[1]
        auc_insertion = torch.sum((insertion_probs[:, 1:] + insertion_probs[:, :-1]) * dx / 2, axis=1)
        cur_avg_auc_insertion = torch.mean(auc_insertion)

        # get the auc of deletion
        dx = 1 / deletion_probs.shape[1]
        auc_deletion = torch.sum((deletion_probs[:, 1:] + deletion_probs[:, :-1]) * dx / 2, axis=1)
        cur_avg_auc_deletion = torch.mean(auc_deletion)

        running_avg_auc_insertion += cur_avg_auc_insertion
        running_avg_auc_deletion += cur_avg_auc_deletion

        num_batches += 1

        torch.cuda.empty_cache()

        avg_auc_insertion = running_avg_auc_insertion / num_batches
        avg_auc_deletion = running_avg_auc_deletion / num_batches

    return avg_auc_insertion, avg_auc_deletion

def main():
    parser = argparse.ArgumentParser(description='Insertion deletion values')

    parser.add_argument('--test_dataset', type=str, default='cifar10', help='Which dataset to train on')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model')
    parser.add_argument('--batch_size', type=int, default=64, help='Testing batch size')
    parser.add_argument('--pixel_batch_size', type=int, default=10, help='Number of pixels to insert/delete at a time')
    parser.add_argument('--sigma', type=float, default=5., help='Sigma of GaussianBlur')
    parser.add_argument('--saliency_method', type=str, default='RISE',
                        help='Which method to use to obtain saliency maps')
    parser.add_argument('--depth', type=int, default=28, help='WRN depth')
    parser.add_argument('--width', type=int, default=2, help='WRN width')
    parser.add_argument('--n_masks', type=int, default=250, help='Number of masks for RISE')
    parser.add_argument('--data_root', type=str, default='../../data/', help='Path to data')
    parser.add_argument('--image_feature_dim', default=64, type=int,
                        help="Dimension of feature maps")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            transforms.Normalize(dataset_statistics[args.test_dataset][0],
                                 dataset_statistics[args.test_dataset][1])
        ])
        testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True,
                                               transform=test_transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif args.test_dataset == 'cifar100':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_statistics[args.test_dataset][0],
                                 dataset_statistics[args.test_dataset][1])
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


    elif args.test_dataset == 'imagenette_160':

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_statistics[args.test_dataset][0],
                                 dataset_statistics[args.test_dataset][1]),
            transforms.Resize((160, 160))
        ])

        testset = ImagenetteDataset(args.data_root + args.test_dataset, 160,
                                    download=True, validation=True, transform=test_transform)

        classes = ("tench", "English springer", "cassette player", "chain saw",
                      "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute")

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    wrn_builder = build_WideResNet(args.depth, args.width, 0.5, image_feature_dim=args.image_feature_dim)
    wrn = wrn_builder.build(classes)
    model = wrn.to(device)

    # replace forward method with get_logits to conform to grad_cam library
    model.forward = model.logits

    # load the model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Successfully loaded model")

    if args.saliency_method == 'RISE':
        generator = RISE(model, input_size=(32, 32), initial_mask_size=(7, 7), n_masks=args.n_masks)
    else:
        generator = None

    blur = GaussianBlur(int(2 * args.sigma - 1), args.sigma)

    avg_auc_insertion, avg_auc_deletion = run_evaluation(testloader, device, model, generator, args.saliency_method,
                                                         blur, args.pixel_batch_size)

    print(f"Average AUC insertion for given model/dataset: {avg_auc_insertion}")
    print(f"Average AUC deletion for given model/dataset: {avg_auc_deletion}")
    print(f"Average AUC insertion - deletion for given model/dataset: {avg_auc_insertion - avg_auc_deletion}")


if __name__ == "__main__":
    main()

