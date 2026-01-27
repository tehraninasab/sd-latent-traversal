import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

import os
from os.path import join
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
WORKSHOP = '/usr/local/data/zahrat/workshop'
sys.path.insert(0, join(WORKSHOP, 'stable-diffusion-finetune'))
sys.path.append(join(WORKSHOP, 'med-classifiers/src'))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from safetensors.torch import load_file
# from ldm.misc_utils import ClipSimilarity
from torchvision import transforms

label_mapping = {
    0: ("nv", "melanocytic nevi"),
    1: ("mel", "melanoma"),
    2: ("bkl", "benign keratosis"),
    3: ("bcc", "basal cell carcinoma"),
    4: ("akiec", "actinic keratosis"),
    5: ("vasc", "vascular lesions"),
    6: ("df", "dermatofibroma"),
}

def load_model(checkpoint_path, device="cuda"):
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
    # Load checkpoint state
    checkpoint = load_file(checkpoint_path)
    # Update UNet weights
    pipe.unet.load_state_dict(checkpoint)
    pipe.to(device)

    return pipe
    
# similarity_metric = ClipSimilarity()

def sample_diffusion(pipe, prompt: str, num_inference_steps=50, guidance_scale=7.5):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    return image
    
# def switch_sampler(model):
#     model.module.sampler = DDIMSampler(model.module.sampler.diffusion_steps, model.module.sampler.noise_schedule, model.module.sampler.beta_schedule, model.module.sampler.loss_type, model.module.sampler.use_fp16, model.module.sampler.use_scale_shift_norm)
#     return model
    
def main(args):
    model = load_model(args.checkpoint_path, args.device)
    transform = transforms.ToTensor()
    os.makedirs(args.output_dir, exist_ok=True)

    dx_dict = {
        "mel": "melanoma",
        "nv": "melanocytic nevi",
        "bkl": "benign keratosis",
        "bcc": "basal cell carcinoma",
        "akiec": "actinic keratosis",
        "vasc": "vascular lesions",
        "df": "dermatofibroma",
    }
    
    classifier_transform = None
    if args.classifier_guided:
        from ham10k_efficientnet import HAM10KClassifier
        
        classifier = HAM10KClassifier()
        checkpoint = torch.load(args.classifier_checkpoint)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(args.device)
        classifier.eval()
        
        classifier_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    for key in dx_dict.keys():
        print(f"Class: {key} - {dx_dict[key]}")
        os.makedirs(os.path.join(args.output_dir, key), exist_ok=True)
        
        ### Sampling Parameters - Start
        guidance_scale = 7.5
        num_inference_steps = 50
        ### Sampling Parameters - End
        
        n_remaining = args.n_samples_per_class
        save_idx = args.start_index
        max_try = 5
        try_num = 0
        while(n_remaining > 0):
            n_remaining -= 1
            prompt = "dermatoscopic image showing " + dx_dict[key]
            
            pred_label_name = None
            max_iter = 5
            iter_count = 0
            while pred_label_name != key and iter_count < max_iter:        
                image = sample_diffusion(model, prompt, num_inference_steps, guidance_scale)            
                tensor_image = classifier_transform(image).unsqueeze(0).to(args.device)
                pred_prob = classifier(tensor_image).softmax(dim=-1).squeeze(0).detach().cpu().numpy()
                pred_label = classifier(tensor_image).softmax(dim=-1).argmax().item()
                pred_label_name = label_mapping[pred_label][0]
                print(f"Predicted label: {pred_label_name}, True label: {key}")
                # if pred_label != key:
                #     model = switch_sampler(model)
                iter_count += 1
                
            if pred_label_name != key and try_num < max_try:
                print(f"Failed to generate image for class {key}")
                print(f"Changing the sampler parameters")
                guidance_scale = guidance_scale + 1
                try_num += 1
            elif pred_label_name != key and try_num == max_try:
                print(f"Failed to generate image for class {key}, skipping to next class")
                break

            elif pred_label_name == key:
                print(f"Saving image {save_idx}")
                save_idx += 1
                image.save(os.path.join(args.output_dir, key, f"generated_image_{save_idx}.png"))    

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize fine-tuned Stable Diffusion model")
    parser.add_argument("--checkpoint_path", type=str, default=join(WORKSHOP, "stable-diffusion-finetune/saved_models/ham10k-finetuned/checkpoint-99-0/model.safetensors"), help="Path to the model checkpoint file")
    # parser.add_argument("--prompt", type=str, default="dermatoscopic image showing melanoma", help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="outputs/ham10000-classifier-guided", help="Path to save the generated image")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--classifier_guided", action="store_true", default=True, help="Use classifier guided sampling")
    parser.add_argument("--classifier_checkpoint", type=str, default=join(WORKSHOP, "med-classifiers/saved_models/7classes/best_model_512_2025-01-30_12-26-51.pth"), help="Path to the classifier checkpoint")
    parser.add_argument("--n_samples_per_class", type=int, default=100, help="Number of samples to generate per class")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for the samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--gpu", type=str, default="1", help="Device to run the model on")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    main(args)

