import torch
from torchvision import datasets, transforms

class ImageNetDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageNet(self.dataset_path, split='train', transform=transform)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        val_dataset = datasets.ImageNet(self.dataset_path, split='val', transform=transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        test_dataset = datasets.ImageNet(self.dataset_path, split='test', transform=transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
