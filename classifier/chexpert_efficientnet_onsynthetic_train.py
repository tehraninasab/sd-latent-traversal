import torch
import os
from os.path import join
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import platform
import torch.distributed as dist
import logging
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import platform
from torch.nn import DataParallel
from tqdm import tqdm
from datasets.chexpert_imagefolder import CheXpertDataset

class CheXpertBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1)
        )
        
    def forward(self, x):
        return self.model(x)

if platform.node() == 'progress':
    PROJECT_ROOT = '/cim/zahrat/workshop/dent'
else:
    PROJECT_ROOT = '/usr/local/data/amarkr/dent'

IMAGE_FOLDER = join(PROJECT_ROOT, 'outputs/chexpert-sd1.5-pe-sd')
SAVE_DIR = join(PROJECT_ROOT, 'saved_models/chexpert-efficientnet-onsynthetic')

BATCH_SIZE = 128
# print('Number of GPUs:', torch.cuda.device_count())
# print("Image root:", IMAGE_ROOT)
# print("Metadata path:", METADATA_PATH)
# print("Project name:", PROJECT_NAME)

def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Configure logging
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler - separate log file for each GPU
    log_file = f'logs/train_{timestamp}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info('='*50)
    logger.info('Starting new training run')
    logger.info(f'Log file: {log_file}')
    logger.info('='*50)
    
    return logger

def setup():
    # Setup logger
    logger = setup_logger()
    logger.info('Initializing training')
    
    # Log system information
    logger.info(f'System: {platform.system()}')
    logger.info(f'Python version: {sys.version}')
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA version: {torch.version.cuda}')
        logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')

def cleanup():
    logger = logging.getLogger('train')
    logger.info('Training completed')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    logger = logging.getLogger('train')
    logger.info('Running training.')
    
    # Create GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels.unsqueeze(1).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()
        
        # Calculate average loss
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                labels = labels.unsqueeze(1).float()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        
        logger.info(
            f'Epoch {epoch+1}/{num_epochs} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}, '
            f'LR: {scheduler.get_last_lr()[0]:.6f}'
        )
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_directory = SAVE_DIR
            os.makedirs(save_directory, exist_ok=True)
            logger.info(f'Saving best model to {save_directory}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Save the inner model
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            },  os.path.join(save_directory, f'best_model_512_{timestamp}.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after epoch {epoch+1}')
                break

def create_datasets(dataset, test_size=0.2):
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        stratify=dataset.labels
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset

def main():
    setup()
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    dataset = CheXpertDataset(image_folder=IMAGE_FOLDER, transform=train_transform)
    print("Dataset length:", len(dataset))
    train_dataset, val_dataset = create_datasets(dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,  # Reduced batch size per GPU
        shuffle=True,
        num_workers=8,  # Reduced workers per GPU
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Create model and move it to GPU
    model = CheXpertBinaryClassifier().cuda()
    model = DataParallel(model)
    
    # Initialize criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Train the model
    try:
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)
    except Exception as e:
        logger = logging.getLogger('train')
        logger.error(f'Error during training: {str(e)}', exc_info=True)
        raise e
    finally:
        cleanup()

if __name__ == '__main__':
    main()
