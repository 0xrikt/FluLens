import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FluLensDataset(Dataset):
    def __init__(self, manifest_path, split="train"):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.rows = []
        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = Path(row["image_path"])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y
