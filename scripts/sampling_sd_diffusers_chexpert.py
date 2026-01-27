import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

import os
from os.path import join
import torch

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
WORKSHOP = '/cim/zahrat/workshop'
sys.path.insert(0, join(WORKSHOP, 'dent'))

import argparse
from safetensors.torch import load_file


def load_model(checkpoint_path, device="cuda"):
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
    # Load checkpoint state
    checkpoint = load_file(checkpoint_path)
    # Update UNet weights
    pipe.unet.load_state_dict(checkpoint)
    pipe = pipe.to(device)
    
    return pipe
    
# similarity_metric = ClipSimilarity()

def sample_diffusion(pipe, prompt: str, num_images_per_prompt, num_inference_steps=50, guidance_scale=7.5):
    with torch.no_grad():
        images = pipe(prompt, num_images_per_prompt=num_images_per_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images

    return images

    
def main(args):
    model = load_model(args.checkpoint_path, args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    conditions = [
            'No Finding', 'Pleural Effusion', 'Support Devices'
        ]
    
    prompts = {
        'normal': "Normal chest X-ray with no significant findings",
        'pe': "Chest X-ray of a 50 year old male subject showing Pleural Effusion",
        'sd': "Chest X-ray of a 50 year old male subject showing Support Devices",
        'pe-sd': "Chest X-ray of a 50 year old male subject showing Pleural Effusion, Support Devices"
    }
    
    for key, prompt in prompts.items():
        print(f"Generating images for {key}...")
        n_remaining = args.n_samples_per_class
        batch_size = args.batch_size
        save_idx = args.start_index
        save_dir = os.path.join(args.output_dir, key)
        os.makedirs(save_dir, exist_ok=True)
        while(n_remaining > 0):
            n_remaining -= batch_size
            images = sample_diffusion(model, prompt, batch_size, args.num_inference_steps)

            for image in images:      
                image.save(os.path.join(save_dir,  f"generated_image_{save_idx}.png"))
                save_idx += 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize fine-tuned Stable Diffusion model")
    parser.add_argument("--checkpoint_path", type=str, default=join(WORKSHOP, "dent/saved_models/finetuned_chexpert-sd1.5/checkpoint-9-2500/model.safetensors"), help="Path to the model checkpoint file")
    # parser.add_argument("--prompt", type=str, default="dermatoscopic image showing melanoma", help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="outputs/chexpert-sd1.5", help="Path to save the generated image")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--classifier_checkpoint", type=str, default=None, help="Path to the classifier checkpoint")
    parser.add_argument("--n_samples_per_class", type=int, default=125, help="Number of samples to generate per class")
    parser.add_argument("--start_index", type=int, default=375, help="Start index for the samples")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for image generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--gpu", type=str, default="7", help="Device to run the model on")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    main(args)

