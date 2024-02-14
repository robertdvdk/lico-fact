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
    def __init__(self, root, split='train', transform=None):
        super(PartImageNetClassificationDataset, self).__init__()
        self.classes = sorted(os.listdir(root + split))
        self.classes_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_images = os.listdir(root + split + '/' + cls)
            self.images.extend(cls_images)
        self.idx_to_images = {i: img for i, img in enumerate(self.images)}
        self.root = root + split + '/'
        self.transform = transform

    def __getitem__(self, item):
        directory = self.root + self.idx_to_images[item].split('_')[0] + '/'
        image = torchvision.io.read_image(directory + self.idx_to_images[item]).float() / 255.
        label = self.classes_to_idx[self.idx_to_images[item].split('_')[0]]

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] == 4:
            image = image[:3]

        if self.transform:
            image = self.transform(image)


        return image, label

    def __len__(self):
        return len(self.images)

class PartImageNetDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(PartImageNetDataset, self).__init__(root, annFile, transform, target_transform, transforms)
    def __getitem__(self, index):
        image, target = super(PartImageNetDataset, self).__getitem__(index)
        print(target)
        print(self.__annotations__)
        try:
            return image, target[0]['category_id']
        except IndexError:
            return image, -1


def split_partimagenet_trainvaltest():
    root = '../../../data/pim/'
    train = root + 'train/'
    val = root + 'val/'
    test = root + 'test/'
    if not os.path.isdir(val):
        os.makedirs(val)
    if not os.path.isdir(test):
        os.makedirs(test)
    for cls in os.listdir(train):
        if not os.path.isdir(val + cls):
            os.makedirs(val + cls)
        if not os.path.isdir(test + cls):
            os.makedirs(test + cls)
        cls_images = os.listdir(train + cls)
        val_images = cls_images[int(0.8 * len(cls_images)):int(0.9 * len(cls_images))]
        test_images = cls_images[int(0.9 * len(cls_images)):]
        for img in val_images:
            os.rename(train + cls + '/' + img, val + cls + '/' + img)
        for img in test_images:
            os.rename(train + cls + '/' + img, test + cls + '/' + img)

if __name__ == "__main__":
    trainset = PartImageNetClassificationDataset('../../../data/pim/', split='train', transform=torchvision.transforms.Resize((224, 224)))
    dl = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    for x, y, z in dl:
        pass
    # split_partimagenet_trainvaltest()