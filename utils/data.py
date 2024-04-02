import os
from pathlib import Path
import wget
import tarfile
from PIL import Image
import torch.utils.data
import torchvision.datasets.coco
import matplotlib.pyplot as plt

class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, root, patch_size=320, download=True, validation=False, transform=None):
        super(ImagenetteDataset, self).__init__()
        if download:
          if not os.path.isdir(root):
            os.makedirs(root)
            url = f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{patch_size}.tgz"
            wget.download(url, out=root)
            with tarfile.open(root+f"/imagenette2-{patch_size}.tgz", 'r:gz') as tar:
                # Extract all contents to the specified directory
                tar.extractall(path=root)
        
        self.folder = Path(root+f'/imagenette2-{patch_size}/train') if not validation else Path(root+f'/imagenette2-{patch_size}/val')
        self.classes = {'n01440764': "tench", 'n02102040': "English springer", 'n02979186': "cassette player", 
                        'n03000684': "chain saw", 'n03028079': "church", 'n03394916': "French horn", 
                        'n03417042': "garbage truck", 'n03425413': "gas pump", 
                        'n03445777': "golf ball", 'n03888257': "parachute"}
        self.classes_to_idx = {
            'n01440764': 0,    # "tench"
            'n02102040': 1,    # "English springer"
            'n02979186': 2,    # "cassette player"
            'n03000684': 3,    # "chain saw"
            'n03028079': 4,    # "church"
            'n03394916': 5,    # "French horn"
            'n03417042': 6,    # "garbage truck"
            'n03425413': 7,    # "gas pump"
            'n03445777': 8,    # "golf ball"
            'n03888257': 9     # "parachute"
        }
        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)
        self.transform = transform
        self.patch_size = patch_size
        self.validation = validation
        
    def __getitem__(self, index):
        image_fname = self.images[index]
        image = Image.open(image_fname).convert('RGB')
        label = image_fname.parent.stem
        label = self.classes_to_idx[label]

        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.images)


class PartImageNetClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, return_masks=False):
        super(PartImageNetClassificationDataset, self).__init__()
        self.root = root
        if not os.path.exists(root + 'class_mapping.txt'):
            self.initialise_dataset()

        self.mask_dir = root + '/annotations/' + split + '/'
        self.img_dir = root + '/images/' + split + '/'
        self.return_masks = return_masks
        self.transform = transform
        self.split = split

        self.images = []
        for image in os.listdir(self.img_dir):
            self.images.append(image)

        self.classid_to_label = {}
        with open(self.root + 'class_mapping.txt') as fopen:
            for line in fopen.readlines():
                idx, _, class_id = line.strip().split('\t')
                self.classid_to_label[class_id] = int(idx)
        self.idx_to_images = {i: img for i, img in enumerate(self.images)}
        self.images_to_idx = {img: i for i, img in enumerate(self.images)}
        print(f'Loaded {len(self.idx_to_images)} images from {self.split} split')

    def __getitem__(self, item):
        image = torchvision.io.read_image(self.img_dir + self.idx_to_images[item]).float() / 255.
        label = self.classid_to_label[self.idx_to_images[item].split('_')[0]]

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] == 4:
            image = image[:3]

        if self.transform:
            image = self.transform(image)

        if self.return_masks:
            return image, label, self.getmasks(item)
        else:
            return image, label

    def initialise_dataset(self):
        imagenet_class_mapping = {}
        with open('./imagenet_classes.txt') as fopen:
            for line in fopen.readlines():
                class_id, _, class_name = line.strip().split(' ')
                class_name = ' '.join(class_name.split('_'))
                imagenet_class_mapping[class_id] = class_name

        classes = set()
        for img in os.listdir(self.root + 'images/train/'):
            class_id = img.split('_')[0]
            classes.add(class_id)

        partimagenet_class_mapping = {imagenet_class_mapping[class_id]: class_id for class_id in classes}

        with open(self.root + 'class_mapping.txt', 'w') as fopen:
            sorted_classnames = sorted(partimagenet_class_mapping.keys(), key=lambda x: x.lower())
            for idx, class_id in enumerate(sorted_classnames):
                fopen.write(f'{idx}\t{class_id}\t{partimagenet_class_mapping[class_id]}\n')

    def __len__(self):
        return len(self.images)

    def getmasks(self, item):
        image_path = self.idx_to_images[item][:-5] + '.png'  # remove .JPEG and add .png
        image = torchvision.io.read_image(self.mask_dir + image_path).float() / 255.
        image = torch.where(image < image.max(), 1, 0)  # make the segmentation mask binary

        if self.transform:
            image = self.transform(image)

        return image

if __name__ == "__main__":
    pass