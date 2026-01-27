import numpy as np
import pandas as pd
from scipy import linalg
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

class ImageFolderDataset(Dataset):
    """Custom dataset for loading images from a folder"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.image_paths = list(self.folder_path.glob('**/*.jpg')) + list(self.folder_path.glob('**/*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_inception_model():
    """Load pretrained InceptionV3 model"""
    model = inception_v3(pretrained=True, transform_input=False)
    # Remove the final classification layer
    model.fc = nn.Identity()
    model.eval()
    return model

def extract_features(dataloader, model, device):
    """Extract features from images using InceptionV3"""
    features = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            # Get features before the final classification layer
            feat = model(batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)

def calculate_fid(features_real, features_fake):
    """
    Calculate Fréchet Inception Distance (FID) between two feature distributions.
    
    Args:
        features_real: numpy array of shape (n_samples, feature_dim) containing real image features
        features_fake: numpy array of shape (m_samples, feature_dim) containing generated image features
    
    Returns:
        fid_score: float representing the FID score
    """
    # Calculate mean and covariance for both distributions
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)
    
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)
    
    # Calculate square root of matrix product
    # First ensure the matrix is Hermitian (symmetric with complex conjugate entries)
    sigma_real = (sigma_real + sigma_real.T) / 2
    sigma_fake = (sigma_fake + sigma_fake.T) / 2
    
    # Calculate sqrt of product using scipy's linalg
    try:
        sqrt_product = linalg.sqrtm(sigma_real @ sigma_fake)
        # Handle numerical errors in complex numbers
        if np.iscomplexobj(sqrt_product):
            if not np.any(np.imag(sqrt_product) > 1e-7):
                sqrt_product = sqrt_product.real
    except ValueError:
        return float('inf')
    
    # Calculate FID score
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    trace_term = np.trace(sigma_real + sigma_fake - 2 * sqrt_product)
    
    fid_score = mean_diff + trace_term
    
    return float(np.real(fid_score))

def compute_fid_from_folders(real_path, fake_path, batch_size=32):
    """
    Compute FID score between two folders of images
    
    Args:
        real_path: path to folder containing real images
        fake_path: path to folder containing generated images
        batch_size: batch size for processing images
    
    Returns:
        fid_score: float representing the FID score
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Initial resize to target size
        transforms.CenterCrop((512, 512)),  # Ensure exact dimensions
        transforms.Resize((299, 299)),  # Resize to InceptionV3 expected size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    real_dataset = ImageFolderDataset(real_path, transform)
    
    fake_dataset = ImageFolderDataset(fake_path, transform)
    
    
    real_loader = DataLoader(real_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Load inception model
    model = get_inception_model().to(device)
    
    # Extract features
    real_features = extract_features(real_loader, model, device)
    fake_features = extract_features(fake_loader, model, device)
    
    # Calculate FID
    return calculate_fid(real_features, fake_features)

if __name__ == "__main__":
    # Example usage
    real_images_path = "/usr/local/faststorage/datasets/CheXpert-v1.0_512x512/train"
    fake_images_path = "/usr/local/data/zahrat/workshop/dent/outputs/chexpert-sd1.5-pe-sd"
    
    
    pe_fid_score = compute_fid_from_folders(
                                            '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_test_images/pe', 
                                            '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_synthetic_images/pe')
    
    sd_fid_score = compute_fid_from_folders(
                                            '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_test_images/sd', 
                                            '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_synthetic_images/sd')
    
    pe_sd_fid_score = compute_fid_from_folders(
                                            '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_test_images/', 
                                            '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_synthetic_images/')
    
    df = pd.DataFrame({'FID Score': [pe_fid_score, sd_fid_score, pe_sd_fid_score]}, index=['PE', 'SD', 'PE + SD'])
    df.to_csv('../metrics/fid_scores.csv')
    # except Exception as e:
    #     print(f"Error computing FID score: {str(e)}")
