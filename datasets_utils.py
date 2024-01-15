import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datasets import load_dataset

class ImageNetDataLoader:
    def __init__(self, dataset_path, streaming=False):
        self.dataset_path = dataset_path
        self.streaming = streaming

    def collate_fn(self, batch):
        images = [x["image"].convert("RGB") for x in batch]
        images = [
            x.resize(
                (256, 256),
            )
            for x in images
        ]
        images = [self.normalize(self.PIL2Tensor(x).to(dtype=torch.float)) for x in images]
        images = torch.stack(images)
        return images

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # TODO: implement shuffling if not streaming
        dataset = load_dataset('imagenet-1k', split='train', data_dir=self.dataset_path, streaming=self.streaming).with_format("torch")
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False if self.streaming else True, collate_fn=self.collate_fn)
        print(next(iter(train_dataloader)))
        val_dataset = load_dataset('imagenet-1k', split='validation', data_dir=self.dataset_path, streaming=self.streaming)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, collate_fn=self.collate_fn)

        test_dataset = load_dataset('imagenet-1k', split='test', data_dir=self.dataset_path, streaming=self.streaming)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, collate_fn=self.collate_fn)

        return train_dataloader, val_dataloader, test_dataloader


class ImageNetDataset(Dataset):
    def __init__(self,
                 data,
                 transforms):

        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]
        image = self.transforms(image)
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.data)