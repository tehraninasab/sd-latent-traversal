import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Import CLASS_TO_IDX from classes.py
from classes import CLASS_TO_IDX, CLASSES

class HAM10KDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = CLASSES

        # Load image paths and labels
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = os.path.join(image_folder, class_name)
            if not os.path.exists(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
