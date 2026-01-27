import os
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(0, '/cim/zahrat/workshop/stable-diffusion-finetune')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from safetensors.torch import load_file
from ldm.misc_utils import ClipSimilarity
from torchvision import transforms

def load_model(config_path, checkpoint_path, device="cuda"):

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model = torch.nn.DataParallel(model)  # Wrap the model for multi-GPU
    model.to(device)
    model.eval()
    return model

def set_image_paths(df, image_root_path):
    image_paths = []

    for index, row in df.iterrows():
        image_id = row['image_id']
        
        base1_path = os.path.join(image_root_path, 'HAM10000_images_part_1')
        base2_path = os.path.join(image_root_path, 'HAM10000_images_part_2')
        
        image_path_1 = os.path.join(base1_path, f'{image_id}.jpg')
        image_path_2 = os.path.join(base2_path, f'{image_id}.jpg')

        if os.path.exists(image_path_1):
            image_paths.append(image_path_1)
        elif os.path.exists(image_path_2):
            image_paths.append(image_path_2)
        else:
            print(f'Image {image_id} not found in either path.')

        # Add the image paths to your DataFrame
    df['image_path'] = image_paths
    return df
    
similarity_metric = ClipSimilarity()

def generate_images(model, prompts, timesteps=500, device="cuda"):
    model.eval()
    with torch.no_grad():
        # Ensure prompts is a list of strings
        if isinstance(prompts, str):
            prompts = [prompts]
        # tokens = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length, padding="max_length")
        # tokens = tokens['input_ids'].to(device)
        conditioning = model.module.get_learned_conditioning(prompts).to(device)  # Access the module for DataParallel

        batch_size = len(prompts)
        latent_shape = (batch_size, model.module.channels, 64, 64)  # Access the module for DataParallel

        # Generate image latents with diffusion
        with torch.no_grad():
            samples = model.module.sample(conditioning, batch_size=batch_size, shape=latent_shape, timesteps=timesteps)  # Access the module for DataParallel

        # Decode latent representation to images
        images = model.module.decode_first_stage(samples)  # Access the module for DataParallel
        images = (images + 1.0) / 2.0
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = (images * 255).astype("uint8")
        return [Image.fromarray(image) for image in images]

def visualize_samples_and_generated_images(df, dx_dict, model, sampler, text_encoder, tokenizer, save_dir):
    fig, axs = plt.subplots(6, len(df.dx.unique()), figsize=(20, 20))

    for i, key in enumerate(df.dx.unique()):
        print(f"Class: {key} - {dx_dict[key]}")
        print(df[df.dx == key].sample(5).image_path.values)
        axs[0, i].set_title(f"{key} - {dx_dict[key]}")
        
        for j, image_path in enumerate(df[df.dx == key].sample(5).image_path.values):
            image = plt.imread(image_path)
            axs[j, i].imshow(image)
            axs[j, i].axis('off')
        
        prompt = f"dermatoscopic image showing {dx_dict[key]}"
        generated_image = generate_image(model, sampler, text_encoder, tokenizer, prompt)
        axs[j + 1, i].imshow(generated_image)
        axs[j + 1, i].axis('off')

    save_path = f"{save_dir}/samples_and_generated_images.png"
    plt.savefig(save_path)

def main(args):
    model = load_model(args.config_path, args.checkpoint_path, args.device)
    transform = transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)

    # df = pd.read_csv(args.csv_file)
    # df = set_image_paths(df, args.image_root_path)
    dx_dict = {
        "mel": "melanoma",
        "nv": "melanocytic nevi",
        "bkl": "benign keratosis",
        "bcc": "basal cell carcinoma",
        "akiec": "actinic keratosis",
        "vasc": "vascular lesions",
        "df": "dermatofibroma",
    }
    for key in dx_dict.keys():
        print(f"Class: {key} - {dx_dict[key]}")
        os.makedirs(os.path.join(args.output_dir, key), exist_ok=True)
        
        n_remaining = args.n_samples_per_class
        save_idx = args.start_index
        while(n_remaining > 0):
            n_prompts = min(args.batch_size, n_remaining)
            n_remaining -= n_prompts
            prompts = ["dermatoscopic image showing " + dx_dict[key]] * n_prompts
            images = generate_images(model, prompts, args.num_steps, args.device)
            for image in images:
                if args.similarity_check:
                    tensor_image = transform(image).unsqueeze(0).to(args.device)
                    image_features = similarity_metric.encode_image(tensor_image)
                    text_features = similarity_metric.encode_text(prompts[0])

                    sim = F.cosine_similarity(image_features, text_features).item()
                
                    if sim >= 0.3:
                        save_idx += 1
                        image.save(os.path.join(args.output_dir, key, f"generated_image_{save_idx}_{sim:.3f}.png")) 
                else:
                    save_idx += 1
                    image.save(os.path.join(args.output_dir, key, f"generated_image_{save_idx}.png"))    
        # visualize_samples_and_generated_images(df, dx_dict, model, sampler, text_encoder, tokenizer, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize fine-tuned Stable Diffusion model")
    parser.add_argument("--config_path", type=str, default="/cim/zahrat/workshop/stable-diffusion-finetune/configs/stable-diffusion/v1-inference.yaml", help="Path to the model config file")
    parser.add_argument("--checkpoint_path", type=str, default="/cim/zahrat/workshop/stable-diffusion-finetune/saved_models/ham10k-finetuned-sd-original/checkpoint-99-0/model.safetensors", help="Path to the model checkpoint file")
    # parser.add_argument("--prompt", type=str, default="dermatoscopic image showing melanoma", help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="outputs/ham10000", help="Path to save the generated image")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of DDIM sampling steps")
    parser.add_argument("--csv_file", type=str, default="/usr/local/faststorage/datasets/zahrat/ham10000/HAM10000_metadata.csv" ,help="Path to the CSV file containing image paths and labels")
    parser.add_argument("--image_root_path", type=str, default="/usr/local/faststorage/datasets/zahrat/ham10000", help="Root path to the images")
    parser.add_argument("--similarity_check", action="store_true", default=False, help="Check similarity between generated image and text prompt")
    parser.add_argument("--n_samples_per_class", type=int, default=100, help="Number of samples to generate per class")
    parser.add_argument("--start_index", type=int, default=120, help="Start index for the samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--gpu", type=str, default="1", help="Device to run the model on")
    
    
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    main(args)


