import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        foldername_to_label = {
            '00': 0,
            '01': 0,
            '10': 1,
            '11': 1
        }

        # Load image paths and labels
        for foldername, labels in foldername_to_label.items():
            folder_path = os.path.join(image_folder, foldername)
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(labels)
                self.class_names.append(foldername)
    
        self.labels = np.array(self.labels).astype(np.float32)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
