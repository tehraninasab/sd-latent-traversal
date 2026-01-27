import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class HAM10KDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_root_path,
        size=None,
        transform=None
    ):
        self.df = csv_file
        self.image_root_path = image_root_path
        self.size = size
        self.transform = transform
        
        self.set_image_labels()
        self.set_image_paths()
        
        
    def set_image_labels(self):   
        """
        "mel": "melanoma",
        "nv": "melanocytic nevi",
        "bkl": "benign keratosis",
        "bcc": "basal cell carcinoma",
        "akiec": "actinic keratosis",
        "vasc": "vascular lesions",
        "df": "dermatofibroma",
        """
            
        self.lesion_type_dict = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df': 6
        }
        
        self.df = self.df.assign(label=self.df['dx'].map(self.lesion_type_dict))

    def set_image_paths(self):
        image_paths = []

        for _, row in self.df.iterrows():
            image_id = row['image_id']
            
            base1_path = os.path.join(self.image_root_path, 'HAM10000_images_part_1')
            base2_path = os.path.join(self.image_root_path, 'HAM10000_images_part_2')
            
            image_path_1 = os.path.join(base1_path, f'{image_id}.jpg')
            image_path_2 = os.path.join(base2_path, f'{image_id}.jpg')

            if os.path.exists(image_path_1):
                image_paths.append(image_path_1)
            elif os.path.exists(image_path_2):
                image_paths.append(image_path_2)
            else:
                print(f'Image {image_id} not found in either path.')

            # Add the image paths to your DataFrame
        self.df = self.df.assign(image_path=image_paths)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root_path, row['image_path'])

        image = Image.open(image_path).convert('RGB')

        if self.size:
            size = (self.size, self.size) if isinstance(self.size, int) else self.size
            image = image.resize(size)      
        if self.transform:
            image = self.transform(image)


        label = row['label']
        
        return image, label

def get_datasets(csv_file: str, image_root_path: str, train_split: float = 0.8, train_transform=None, eval_transform=None, size=None, seed=42):
    df = pd.read_csv(csv_file)
    
    # Split the df into train, test, and val df
    df['split'] = None
    unique_lesion_ids = list(df['lesion_id'].unique())
    lesion_id_to_dx = df.set_index('lesion_id')['dx'].to_dict()
    
    train_lesion_ids, temp_lesion_ids = train_test_split(
        unique_lesion_ids, train_size=train_split, random_state=seed, 
        stratify=[lesion_id_to_dx[lesion_id] for lesion_id in unique_lesion_ids])
    
    val_lesion_ids, test_lesion_ids = train_test_split(
        temp_lesion_ids, test_size=0.5, random_state=seed, 
        stratify=[lesion_id_to_dx[lesion_id] for lesion_id in temp_lesion_ids])
    
    df.loc[df['lesion_id'].isin(train_lesion_ids), 'split'] = 'train'
    df.loc[df['lesion_id'].isin(val_lesion_ids), 'split'] = 'val'
    df.loc[df['lesion_id'].isin(test_lesion_ids), 'split'] = 'test'
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print(f'Train samples: {len(train_df)}')
    print(f'Val samples: {len(val_df)}')
    print(f'Test samples: {len(test_df)}')
    
    train_dataset = HAM10KDataset(train_df, image_root_path, size=size, transform=train_transform)
    val_dataset = HAM10KDataset(val_df, image_root_path, size=size, transform=eval_transform)
    test_dataset = HAM10KDataset(test_df, image_root_path, size=size, transform=eval_transform)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_datasets(
        '/usr/local/faststorage/datasets/zahrat/ham10000/HAM10000_metadata.csv',
        '/usr/local/faststorage/datasets/zahrat/ham10000',
        train_transform=transforms.ToTensor(),
        eval_transform=transforms.ToTensor(),
        size=512,
        seed=42
    )
    
    print(train_dataset[0][0].shape, train_dataset[0][1])
