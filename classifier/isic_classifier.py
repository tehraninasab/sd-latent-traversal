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
import sys, socket
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# Configuration
CHECKPOINT_DIR = "./saved_models/isic-efficientnet/mel_nv_classifier.pt"
os.makedirs(os.path.dirname(CHECKPOINT_DIR), exist_ok=True)
if socket.gethostname() == 'inferentia':
    TRAIN_CSV = '/home/amarkr/dent/data/isic2019/isic2019_train.csv'  
    VAL_CSV = '/home/amarkr/dent/data/isic2019/isic2019_val.csv'
    TEST_CSV = '/home/amarkr/dent/data/isic2019/isic2019_test.csv'  # Added test CSV path

else:
    TRAIN_CSV = 'data/isic2019/isic2019_train.csv'  
    VAL_CSV = 'data/isic2019/isic2019_val.csv'
    TEST_CSV = 'data/isic2019/isic2019_test.csv'  
BATCH_SIZE = 128  
IMAGE_SIZE = 512 

def parse_args():
    parser = argparse.ArgumentParser(description='ISIC Classifier Training/Testing Script')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                      help='Mode to run the model in (train or test, default: train)')
    parser.add_argument('--checkpoint', type=str, default='saved_models/isic-efficientnet/isic_disease_classifier.pth',
                      help='Path to model checkpoint for testing')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of epochs for training')
    return parser.parse_args()

def setup_logger(rank, mode):
    if rank != 0:
        return None
        
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    logger = logging.getLogger('isic_classifier')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    log_file = f'classifier/logs/isic_{mode}_foura100_{timestamp}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info('='*50)
    logger.info(f'Starting new ISIC {mode} run')
    logger.info(f'Log file: {log_file}')
    logger.info('='*50)
    
    return logger

def setup(rank, world_size, mode):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    logger = setup_logger(rank, mode)
    
    if rank == 0:
        logger.info(f'Initializing process group with {world_size} GPUs')
        logger.info(f'System: {platform.system()}')
        logger.info(f'Python version: {sys.version}')
        logger.info(f'PyTorch version: {torch.__version__}')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            logger.info(f'CUDA version: {torch.version.cuda}')
            logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    
    return logger

def cleanup(rank, logger):
    if rank == 0 and logger:
        logger.info('Training completed')
    dist.destroy_process_group()

class ISICDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[(self.data['MEL'] == 1) | (self.data['NV'] == 1)]
        self.data['label'] = self.data['MEL'].astype(int)
        self.transform = transform
        if socket.gethostname() == 'four-a100':
            self.base_path = '/home/jupyter/ISIC_2019_Training_Input/orig'
        else:
            self.base_path = '/mnt/l4-data/ISIC_2019_Training_Input/orig'
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.base_path, self.data.iloc[idx]['image'] + '.jpg')
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            img_name = os.path.join(self.base_path, self.data.iloc[idx]['image'] + '_downsampled.jpg')
            image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        return image, label

class ISICClassifier(nn.Module):
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

def train_model(rank, world_size, model, train_loader, val_loader, criterion, optimizer, scheduler, logger, num_epochs=20):
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
            predicted = torch.sigmoid(outputs) > 0.5
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            if batch_idx % 50 == 0 and rank == 0 and logger:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                          f'Acc: {100 * train_correct/train_total:.2f}%, '
                          f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        dist.all_reduce(torch.tensor(train_loss).to(rank))
        train_loss = train_loss / world_size
        
        if rank == 0 and logger:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(rank), labels.to(rank)
                    labels = labels.unsqueeze(1)
                    
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = torch.sigmoid(outputs) > 0.5
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100 * val_correct / val_total
            
            logger.info(
                f'Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_accuracy:.2f}%, '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )
            
            if val_loss < best_val_loss:
                logger.info("Saving checkpoint...")
                best_val_loss = val_loss
                patience_counter = 0
                save_path = join(os.path.dirname(CHECKPOINT_DIR), 
                               f'isic_disease_classifier.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after epoch {epoch+1}')
                    break
        
        dist.barrier()

def test_model(rank, model, test_loader, criterion, logger):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            labels = labels.unsqueeze(1)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            predicted = probs > 0.5
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * test_correct / test_total
    
    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc_score = roc_auc_score(all_labels, all_preds)
    avg_precision = average_precision_score(all_labels, all_preds)
    
    if logger:
        logger.info(f'Test Results:')
        logger.info(f'Loss: {test_loss:.4f}')
        logger.info(f'Accuracy: {test_accuracy:.2f}%')
        logger.info(f'AUC-ROC: {auc_score:.4f}')
        logger.info(f'Average Precision: {avg_precision:.4f}')
    
    return test_loss, test_accuracy, auc_score, avg_precision

def main_worker(rank, world_size, args):
    logger = setup_logger(rank, args.mode)
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.mode == 'train':
        setup(rank, world_size, args.mode)
        
        train_dataset = ISICDataset(TRAIN_CSV, transform=train_transform)
        val_dataset = ISICDataset(VAL_CSV, transform=test_transform)
        
        train_sampler = DistributedSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=4,
            pin_memory=True
        ) if rank == 0 else None
        
        model = ISICClassifier().to(rank)
        model = DDP(model, device_ids=[rank])
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
        try:
            train_model(rank, world_size, model, train_loader, val_loader, criterion, 
                       optimizer, scheduler, logger, num_epochs=args.num_epochs)
        finally:
            cleanup(rank, logger)
            
    elif args.mode == 'test':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for test mode")
        
        # For testing, we only use one GPU
        if rank == 0:
            test_dataset = ISICDataset(TEST_CSV, transform=test_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                num_workers=4,
                pin_memory=True
            )
            
            model = ISICClassifier().to(rank)
            checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{rank}')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            criterion = nn.BCEWithLogitsLoss()
            
            logger.info(f'Loading checkpoint from {args.checkpoint}')
            logger.info(f'Checkpoint epoch: {checkpoint["epoch"]}')
            logger.info(f'Validation loss: {checkpoint["val_loss"]:.4f}')
            logger.info(f'Validation accuracy: {checkpoint["val_accuracy"]:.2f}%')
            
            test_loss, test_accuracy, auc_score, avg_precision = test_model(
                rank, model, test_loader, criterion, logger)

if __name__ == '__main__':
    args = parse_args()
    world_size = 1#torch.cuda.device_count() if args.mode == 'train' else 1
    torch.multiprocessing.spawn(main_worker,
                              args=(world_size, args),
                              nprocs=world_size,
                              join=True)