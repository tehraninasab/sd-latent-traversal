from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image
import torch
import matplotlib.pyplot as plt
from scipy.special import comb
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def get_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_A = np.linalg.norm(v1)
    norm_B = np.linalg.norm(v2)

    # if norm_A == 0 or norm_B == 0:  # Prevent division by zero
    #     return False  

    cosine_similarity = dot_product / (norm_A * norm_B)
    return cosine_similarity

def latents_to_pil(latents, vae):
    latents = 1 / 0.18215 * latents  # Scale latents back
    image = vae.decode(latents).sample  # Decode latents to pixel space
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    image = (image * 255).byte().cpu().numpy()  # Convert to [0,255] and NumPy
    image = image.transpose(0, 2, 3, 1)  # Rearrange (B, C, H, W) → (B, H, W, C)
    pil_images = [Image.fromarray(img) for img in image]  # Convert to PIL
    return pil_images

from os.path import join
import torch

img_root = '/usr/local/data/zahrat/workshop/dent_output/chexpert-swap-normal/support_devices'

patient_ids = list(range(1500, 3350))

# pe_imgs = {pid: [join(img_root, f'seed{pid}/'normal-pleural_effusion/lambda_t_star{lambda_t_star}/style.png') for lambda_t_star in range(0, 31, 5)] for pid in patient_ids}

sd_img_paths = {pid: [join(img_root, f'seed{pid}/lambda_t_star{5}/neutral.png')] + [join(img_root, f'seed{pid}/lambda_t_star{lambda_t_star}/style.png') for lambda_t_star in range(45, 4, -5)] for pid in patient_ids}
sd_latent_paths = {pid: [join(img_root, f'seed{pid}/lambda_t_star{5}/neutral.pt')] + [join(img_root, f'seed{pid}/lambda_t_star{lambda_t_star}/style.pt') for lambda_t_star in range(45, 4, -5)] for pid in patient_ids}

sd_latents = {}
for pid, paths in sd_latent_paths.items():
    sd_latents[pid] = [torch.load(path, map_location='cpu').reshape(-1).cpu().numpy() for path in paths]
    

img_root = '/usr/local/data/zahrat/workshop/dent_output/chexpert-swap-normal-negative-prompt/pleural_effusion'

patient_ids = list(range(1500, 4000))

pe_img_paths = {pid: [join(img_root, f'seed{pid}/lambda_t_star{5}/neutral.png')] + [join(img_root, f'seed{pid}/lambda_t_star{lambda_t_star}/style.png') for lambda_t_star in range(45, 4, -5)] for pid in patient_ids}
pe_latent_paths = {pid: [join(img_root, f'seed{pid}/lambda_t_star{5}/neutral.pt')] + [join(img_root, f'seed{pid}/lambda_t_star{lambda_t_star}/style.pt') for lambda_t_star in range(45, 4, -5)] for pid in patient_ids}

pe_latents = {}
for pid, paths in pe_latent_paths.items():
    pe_latents[pid] = [torch.load(path, map_location='cpu').reshape(-1).cpu().numpy() for path in paths]
    
    
checkpoint_path = '../saved_models/finetuned_chexpert-sd1.5/checkpoint-9-2500/model.safetensors'
device = 'cuda'

model = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            safety_checker=None,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

if checkpoint_path:
    checkpoint = load_file(checkpoint_path)
    model.unet.load_state_dict(checkpoint)

model = model.to(device)
model.to(device)

def bezier_curve(t, points):
    """Computes a Bézier curve interpolation."""
    n = len(points) - 1
    curve = np.zeros_like(points[0])
    
    for i in range(n + 1):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += bernstein * points[i]
    
    return curve

save_dir = '/usr/local/data/zahrat/workshop/dent_output/interpolations/support_devices'
os.makedirs(save_dir, exist_ok=True)

for pid, latents_1d_sd in tqdm(sd_latents.items()):

    # Example: Using Bézier interpolation for 7 latent vectors
    latent_vectors = latents_1d_sd

    num_steps = 50  # Number of steps along the curve

    for i, t in enumerate(np.linspace(0, 1, num_steps)):
        v = bezier_curve(t, latent_vectors)
        os.makedirs(join(save_dir, f'seed{pid}'), exist_ok=True)
    
        image = latents_to_pil(torch.tensor(v.reshape((1, 4, 64, 64))).cuda(), model.vae)[0]
        image.save(join(save_dir, f'seed{pid}', f'bezier_{pid}_{i}.png'))

save_dir = '/usr/local/data/zahrat/workshop/dent_output/interpolations/pleural_effusion'
os.makedirs(save_dir, exist_ok=True)
    
for pid, latents_1d_pe in tqdm(pe_latents.items()):

    # Example: Using Bézier interpolation for 7 latent vectors
    latent_vectors = latents_1d_pe

    num_steps = 50  # Number of steps along the curve

    for i, t in enumerate(np.linspace(0, 1, num_steps)):
        v = bezier_curve(t, latent_vectors)
        os.makedirs(join(save_dir, f'seed{pid}'), exist_ok=True)
    
        image = latents_to_pil(torch.tensor(v.reshape((1, 4, 64, 64))).cuda(), model.vae)[0]
        image.save(join(save_dir, f'seed{pid}', f'bezier_{pid}_{i}.png'))
