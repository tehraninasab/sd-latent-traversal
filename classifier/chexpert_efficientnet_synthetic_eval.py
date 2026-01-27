import torch
import torch.nn as nn
import os
from os.path import join
import shutil
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
from torchvision import transforms
from chexpert_efficientnet import CheXpertClassifier
from datasets.chexpert_imagefolder import CheXpertDataset

from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torchvision.datasets import ImageFolder
import platform

if platform.node() == 'progress':
    PROJECT_ROOT = '/cim/zahrat/workshop/dent'
else:   
    PROJECT_ROOT = '/usr/local/data/zahrat/workshop/dent'
    
IMAGE_FOLDER = join(PROJECT_ROOT, 'outputs/chexpert-sd1.5')
SAVE_FILTERED = join(PROJECT_ROOT, 'outputs/chexpert-sd1.5-filtered')
CHECKPOINT_PATH = join(PROJECT_ROOT, 'saved_models/chexpert_efficientnet_onreal/best_model_512_2025-02-09_11-34-09.pth')
SAVE_DIR = join(PROJECT_ROOT, 'outputs/chexpert-efficientnet-synthetic')
NUM_CLASSES = 2
CLASS_NAMES = ['Pleural Effusion', 'Support Devices']

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_raw_outputs = []  # Store raw probabilities for ROC-AUC
    
    # Use tqdm for progress bar
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Use mixed precision inference
            with autocast():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.nn.functional.one_hot(probs.argmax(dim=1), num_classes=NUM_CLASSES)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_raw_outputs.append(probs.cpu().numpy())
    
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_raw_outputs))

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics including ROC-AUC"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                             average='binary', 
                                                             zero_division=0)
    
    # Handle cases where a class might have all negative samples
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except:
        auc_score = np.nan
        
    return accuracy, precision, recall, f1, auc_score

def plot_confusion_matrices(all_labels, all_preds, class_names, save_dir):
    """Plot and save confusion matrices for each class"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_{class_name}_{timestamp}.png')
        plt.close()

def analyze_prediction_distribution(all_raw_outputs, class_names, save_dir):
    """Analyze and plot prediction probability distributions"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        sns.kdeplot(all_raw_outputs[:, i], label=class_name)
    
    plt.title('Prediction Probability Distributions')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_distributions_{timestamp}.png')
    plt.close()

def save_correctly_classified_samples(test_dataset, all_preds, all_labels, save_dir):
    """Save correctly classified samples to a specified directory"""
    os.makedirs(save_dir, exist_ok=True)
    correct_indices = np.where((all_preds == all_labels).all(axis=1))[0]
    print(f"Found {len(correct_indices)} correctly classified samples.")
    for idx in correct_indices:
        img_path = test_dataset.image_paths[idx]
        class_name = test_dataset.class_names[idx]
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        save_path = os.path.join(class_dir, img_name)
        # copy the image to the new directory
        print(f"Copying {img_path} to {save_path}")
        shutil.copy(img_path, save_path)

def main():
    # Set device and enable cuDNN benchmarking
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Load model
    model = CheXpertClassifier()
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Data loading with optimized settings
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the test dataset using ImageFolder
    test_dataset = CheXpertDataset(image_folder=IMAGE_FOLDER, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8, 
                           pin_memory=True, shuffle=False)

    # Evaluate model
    print("Starting model evaluation...")
    all_preds, all_labels, all_raw_outputs = evaluate_model(model, test_loader, device)
    # Calculate metrics for each class
    results = []

    for i, class_name in enumerate(CLASS_NAMES):

        accuracy, precision, recall, f1, auc = calculate_metrics(
            all_labels[:, i], all_preds[:, i], all_raw_outputs[:, i]
        )
        results.append({
            'Class': class_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{SAVE_DIR}/metrics_{timestamp}.csv', index=False)
    
    # Print results
    print("\nDetailed Results:")
    print(df_results.to_string(index=False))
    # Calculate and print average metrics
    avg_metrics = df_results.drop('Class', axis=1).mean()
    print("\nAverage Metrics:")
    print(avg_metrics.to_string())
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrices(all_labels, all_preds, CLASS_NAMES, SAVE_DIR)
    analyze_prediction_distribution(all_raw_outputs, CLASS_NAMES, SAVE_DIR)
    
    # Save correctly classified samples
    save_correctly_classified_samples(test_dataset, all_preds, all_labels, SAVE_FILTERED)

    # Confusion matrix for all classes
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(all_labels.argmax(axis=1), all_preds.argmax(axis=1), normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=CLASS_NAMES,
               yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - All Classes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/confusion_matrix_all_{timestamp}.png')
    plt.close()
    
    # Calculate per-class prediction times
    print("\nAnalyzing class-wise correlations...")
    correlation_matrix = np.corrcoef(all_raw_outputs.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Class Prediction Correlations')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/class_correlations_{timestamp}.png')
    plt.close()

if __name__ == '__main__':
    main()
