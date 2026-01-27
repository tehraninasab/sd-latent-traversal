import torch
from os.path import join, exists
import os
import pandas as pd
import numpy as np
import math
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compute LPIPS using multiple GPUs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--total-gpus', type=int, default=4, help='Total number of GPUs')
    parser.add_argument('--disease', type=str, default='melanoma', help='Disease type')
    parser.add_argument('--style', type=str, default='hairs', help='Style type')
    args = parser.parse_args()
    
    # Set the GPU for this process
    gpu_id = args.gpu
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using GPU {gpu_id} for {args.disease}/{args.style}")
    
    # Import necessary modules (do this after setting GPU to avoid issues)
    import lpips
    from PIL import Image
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    
    # Define the transform for images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Initialize LPIPS model on this GPU
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Define dataset
    class LPIPSDataset(Dataset):
        def __init__(self, neutral_path, styled_paths):
            self.neutral_path = neutral_path
            self.styled_paths = styled_paths
            self.neutral_img = None
            
        def __len__(self):
            return len(self.styled_paths)
        
        def __getitem__(self, idx):
            try:
                # Load neutral image (only once)
                if self.neutral_img is None:
                    self.neutral_img = self.load_image(self.neutral_path)
                
                # Load styled image
                styled_img = self.load_image(self.styled_paths[idx])
                
                # Get the step number for reference
                step = int(self.styled_paths[idx].split('_')[-1].split('.')[0])
                
                return self.neutral_img, styled_img, step
            except Exception as e:
                # Return None for problematic images
                return None, None, -1
        
        def load_image(self, path):
            if exists(path):
                img = Image.open(path).convert('RGB')
                return transform(img)
            return None
    
    # Set paths
    img_root = f'/home/jupyter/dent_outputs/interpolations/{args.disease}/{args.style}'
    
    # Check if directory exists
    if not os.path.exists(img_root):
        print(f"Directory {img_root} not found. Exiting...")
        return
    
    # Get all existing patient IDs
    try:
        available_seeds = [int(d.replace('seed', '')) for d in os.listdir(img_root) 
                          if os.path.isdir(join(img_root, d)) and d.startswith('seed')]
        if not available_seeds:
            print(f"No seed directories found in {img_root}. Exiting...")
            return
        all_patient_ids = sorted(available_seeds)
    except Exception as e:
        print(f"Error listing directories in {img_root}: {e}")
        # Fallback to specified range
        all_patient_ids = list(range(4000, 5000))
    
    # Distribute patients to this GPU
    total_patients = len(all_patient_ids)
    patients_per_gpu = math.ceil(total_patients / args.total_gpus)
    start_idx = gpu_id * patients_per_gpu
    end_idx = min(start_idx + patients_per_gpu, total_patients)
    patient_ids = all_patient_ids[start_idx:end_idx]
    
    print(f"GPU {gpu_id} processing {len(patient_ids)} patients ({start_idx}-{end_idx-1}) out of {total_patients}")
    
    # Process each patient
    patient_results = []
    for pid in tqdm(patient_ids, desc=f"GPU {gpu_id}"):
        neutral_path = join(img_root, f'seed{pid}/bezier_{pid}_0.png')
        
        # Skip if neutral image doesn't exist
        if not exists(neutral_path):
            continue
        
        # Define styled image paths (skip the neutral image)
        styled_paths = [
            join(img_root, f'seed{pid}/bezier_{pid}_{step}.png') 
            for step in range(1, 50)  # Skip 0 (neutral)
        ]
        
        # Filter only existing paths
        styled_paths = [path for path in styled_paths if exists(path)]
        
        if not styled_paths:
            continue
        
        try:
            # Create dataset and dataloader
            dataset = LPIPSDataset(neutral_path, styled_paths)
            dataloader = DataLoader(dataset, batch_size=16, num_workers=2)
            
            # Store results for this patient
            lpips_scores = {}
            
            # Process batches
            for batch in dataloader:
                neutral_imgs, styled_imgs, steps = batch
                
                # Filter out None values
                valid_samples = [(n, s, st.item()) for n, s, st in 
                               zip(neutral_imgs, styled_imgs, steps) 
                               if n is not None and s is not None]
                
                if not valid_samples:
                    continue
                    
                neutral_batch, styled_batch, valid_steps = zip(*valid_samples)
                
                # Stack tensors
                neutral_batch = torch.stack(neutral_batch).to(device)
                styled_batch = torch.stack(styled_batch).to(device)
                
                # Compute LPIPS
                with torch.no_grad():
                    distances = loss_fn(neutral_batch, styled_batch)
                    
                # Store results
                for dist, step in zip(distances, valid_steps):
                    lpips_scores[f'step_{step}'] = float(dist.cpu().numpy())
            
            # Add patient record
            patient_record = {
                'pid': pid,
                **lpips_scores
            }
            patient_results.append(patient_record)
            
            # Clear cache periodically
            if len(patient_results) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing patient {pid}: {e}")
            continue
    
    # Save partial results from this GPU
    if patient_results:
        output_dir = '../metrics'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/lpips_{args.disease}_{args.style}_gpu{gpu_id}.csv'
        pd.DataFrame(patient_results).to_csv(output_path, index=False)
        print(f"GPU {gpu_id} saved {len(patient_results)} results to {output_path}")
    else:
        print(f"GPU {gpu_id} found no valid results to save")

if __name__ == "__main__":
    main()