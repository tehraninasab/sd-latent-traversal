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

# CLASS_NAME = 'Pleural Effusion'
CLASS_NAME = 'Support Devices'
CHECKPOINT_DIR = f"./saved_models/chexpert-efficientnet/binary_classifier_{'-'.join(CLASS_NAME.lower().split(' '))}.pt"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TRAIN_SPLIT = './data/chexpert/chexpert_train.csv'
VAL_SPLIT = './data/chexpert/chexpert_val.csv'
BATCH_SIZE = 128

def setup_logger(rank):
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Configure logging
    logger = logging.getLogger(f'GPU{rank}')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler - separate log file for each GPU
    log_file = f'logs/gpu{rank}_{timestamp}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler - only for rank 0
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    if rank == 0:
        logger.info('='*50)
        logger.info('Starting new training run')
        logger.info(f'Log file: {log_file}')
        logger.info('='*50)
    
    return logger

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Changed port number
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Setup logger
    logger = setup_logger(rank)
    logger.info(f'Initializing process group on rank {rank}')
    
    # Log system information on rank 0
    if rank == 0:
        logger.info(f'System: {platform.system()}')
        logger.info(f'Python version: {sys.version}')
        logger.info(f'PyTorch version: {torch.__version__}')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            logger.info(f'CUDA version: {torch.version.cuda}')
            logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')
                
def cleanup(rank):
    logger = logging.getLogger(f'GPU{rank}')
    logger.info('Cleaning up distributed process group')
    dist.destroy_process_group()
    
    if rank == 0:
        logger.info('Training completed')


class CheXpertDataset(Dataset):
    def __init__(self, csv_file, class_name, transform=None):
        data = pd.read_csv(csv_file)
        
        required_columns = ['Path', class_name]
        data = data[required_columns]
        
        print(f'Original {class_name} distribution:')
        print(data[class_name].value_counts())    
            
        self.data = data
        
        self.transform = transform
        if platform.node() == 'progress':
            base_path = '/cim/data/CheXpert-v1.0_512x512/'
        else:
            base_path = '/usr/local/faststorage/datasets/CheXpert-v1.0_512x512/'
        self.image_paths = [os.path.join(base_path, '/'.join(path.split('/')[1:])) 
                          for path in self.data.iloc[:, 0]]
        self.labels = torch.tensor(self.data.iloc[:, 1].values.astype(np.float32))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
    
    
class CheXpertClassifier(nn.Module):
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


def train_model(rank, world_size, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    logger = logging.getLogger(f'GPU{rank}')
    logger.info(f'Running training on rank {rank}.')
    
    # Create GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
            if batch_idx % 100 == 0 and rank == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()
        
        # Calculate average loss across all processes
        train_loss = train_loss / len(train_loader.dataset)
        dist.all_reduce(torch.tensor(train_loss).to(rank))
        train_loss = train_loss / world_size
        
        # Validation phase (only on rank 0 to avoid redundant computation)
        if rank == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(rank), labels.to(rank)
                    labels = labels.unsqueeze(1)
                    
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
                print("Saving checkpoint ...")
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if rank == 0:
                    save_path = join(CHECKPOINT_DIR, f'best_model_512_{timestamp}.pth')
                    print(f'Saving model to {save_path}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),  # Save the inner model
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    },   save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after epoch {epoch+1}')
                    break
        
        # Make sure all processes wait for validation to complete
        dist.barrier()
        
        
def main_worker(rank, world_size):
    setup(rank, world_size)
    
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
    train_dataset = CheXpertDataset(TRAIN_SPLIT, CLASS_NAME,
                                  transform=train_transform)
    val_dataset = CheXpertDataset(VAL_SPLIT, CLASS_NAME,
                                transform=val_transform)
    
    # Create distributed sampler for training data
    train_sampler = DistributedSampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,  # Reduced batch size per GPU
        sampler=train_sampler,
        num_workers=4,  # Reduced workers per GPU
        pin_memory=True
    )
    
    # Validation loader doesn't need to be distributed
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4,
        pin_memory=True
    ) if rank == 0 else None
    
    # Create model and move it to GPU
    model = CheXpertClassifier().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Initialize criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Train the model
    try:
        train_model(rank, world_size, model, train_loader, val_loader, criterion, 
                   optimizer, scheduler, num_epochs=10)
    except Exception as e:
        logger = logging.getLogger(f'GPU{rank}')
        logger.error(f'Error during training: {str(e)}', exc_info=True)
        raise e
    finally:
        cleanup(rank)
        
if __name__ == '__main__':
    world_size = 1  # Number of GPUs
    torch.multiprocessing.spawn(main_worker,
                              args=(world_size,),
                              nprocs=world_size,
                              join=True)
