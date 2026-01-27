import os
from os.path import join
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
import torch.multiprocessing as mp
from PIL import Image
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from scipy.special import comb
from tqdm import tqdm

def get_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_A = np.linalg.norm(v1)
    norm_B = np.linalg.norm(v2)
    cosine_similarity = dot_product / (norm_A * norm_B)
    return cosine_similarity

def latents_to_pil(latents, vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).byte().cpu().numpy()
    image = image.transpose(0, 2, 3, 1)
    pil_images = [Image.fromarray(img) for img in image]
    return pil_images

def bezier_curve(t, points):
    n = len(points) - 1
    curve = np.zeros_like(points[0])
    for i in range(n + 1):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += bernstein * points[i]
    return curve

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_batch(rank, world_size, disease, artifact, patient_ids):
    setup(rank, world_size)
    
    # Initialize model on current GPU
    device = torch.device(f'cuda:{rank}')
    checkpoint_path = '/home/amarkr/dent/saved_models/isic2019_finetune_sd1.5/checkpoint-300-0/model.safetensors'
    
    model = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        safety_checker=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).to(device)

    if checkpoint_path:
        checkpoint = load_file(checkpoint_path)
        model.unet.load_state_dict(checkpoint)

    # Divide work among GPUs
    chunk_size = len(patient_ids) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else len(patient_ids)
    local_patient_ids = patient_ids[start_idx:end_idx]

    img_root = f'/home/jupyter/dent_outputs/isic2019_v2/{disease}/{artifact}'
    save_dir = f'/home/jupyter/dent_outputs/interpolations/{disease}/{artifact}'
    os.makedirs(save_dir, exist_ok=True)

    # Process assigned patients
    for pid in tqdm(local_patient_ids, desc=f'GPU {rank} processing'):
        latent_paths = [join(img_root, f'seed{pid}/lambda_t_star{5}/neutral.pt')] + \
                      [join(img_root, f'seed{pid}/lambda_t_star{lambda_t_star}/style.pt') 
                       for lambda_t_star in range(45, 4, -5)]
        
        try:
            latents_1d_sd = [torch.load(path, map_location=device).reshape(-1).cpu().numpy() 
                            for path in latent_paths]
        except FileNotFoundError:
            print(f'Files not found for patient {pid} on GPU {rank}')
            continue

        os.makedirs(join(save_dir, f'seed{pid}'), exist_ok=True)
        
        # Generate interpolations
        num_steps = 50
        for i, t in enumerate(np.linspace(0, 1, num_steps)):
            v = bezier_curve(t, latents_1d_sd)
            latent_tensor = torch.tensor(v.reshape((1, 4, 64, 64))).to(device)
            image = latents_to_pil(latent_tensor, model.vae)[0]
            image.save(join(save_dir, f'seed{pid}', f'bezier_{pid}_{i}.png'))

    cleanup()

def main():
    world_size = 4  # Number of GPUs
    diseases = ['melanoma', 'nevus']
    artifacts = ['gel_bubbles', 'hairs', 'ink', 'ruler']
    
    for disease in diseases:
        for artifact in artifacts:
            print(f'Processing {disease}:{artifact}')
            patient_ids = list(range(4000, 5000))
            
            # Start multi-GPU processing
            mp.spawn(
                process_batch,
                args=(world_size, disease, artifact, patient_ids),
                nprocs=world_size,
                join=True
            )

if __name__ == "__main__":
    main()