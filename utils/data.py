import os
from pathlib import Path
import wget
import tarfile
import torchvision
from PIL import Image
from torchvision import transforms

class ImagenetteDataset(object):
    def __init__(self, root, patch_size=320, download=True, validation=False, transform=None):
        if download:
          if not os.path.isdir(root):
            os.makedirs(root)
            url = f"https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{patch_size}.tgz"
            filename = wget.download(url, out=root)
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

        #if image.shape[0] == 1: image = image.expand(3, 320, 320)
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.images)