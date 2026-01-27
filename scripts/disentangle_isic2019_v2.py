import os
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
from pytorch_lightning import seed_everything
from dataclasses import dataclass
from typing import Optional, Tuple
import gc
from torch import optim
import torchvision
import clip, socket
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import torch.multiprocessing as mp


# Keep the VGGPerceptualLoss and DCLIPLoss classes exactly the same
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class DCLIPLoss(torch.nn.Module):
    def __init__(self):
        super(DCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=16)

    def forward(self, image1, image2, text1, text2):
        text1 = clip.tokenize([text1]).to("cuda")
        text2 = clip.tokenize([text2]).to("cuda")
        image1 = image1.unsqueeze(0).cuda()
        image2 = image2.unsqueeze(0).cuda()
        image1 = self.avg_pool(self.upsample(image1))
        image2 = self.avg_pool(self.upsample(image2))
        image1_feat = self.model.encode_image(image1)
        image2_feat = self.model.encode_image(image2)
        text1_feat = self.model.encode_text(text1)
        text2_feat = self.model.encode_text(text2)
        d_image_feat = image1_feat - image2_feat
        d_text_feat = text1_feat - text2_feat
        similarity = torch.nn.CosineSimilarity()(d_image_feat, d_text_feat)
        return 1 - similarity

perceptual_loss = VGGPerceptualLoss().cuda()
clip_loss = DCLIPLoss().cuda()

@dataclass
class InferenceConfig:
    checkpoint_path: Optional[str] = None
    device: str = 'cuda'
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    attention_slicing_size: int = 4
    lambda_t_default_1 = 1.0
    lambda_t_default_2 = 0.0
    progress_bar: bool = False

perceptual_loss = VGGPerceptualLoss().cuda()
clip_loss = DCLIPLoss().cuda()

@dataclass
class InferenceConfig:
    checkpoint_path: Optional[str] = None
    device: str = 'cuda'
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    attention_slicing_size: int = 4  # Added for memory optimization
    lambda_t_default_1 = 1.0
    lambda_t_default_2 = 0.0
    progress_bar: bool = False

class StableDiffusionInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Set memory optimization configs
        torch.cuda.empty_cache()
        gc.collect()
        
        # Initialize model with memory optimizations
        self.model = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            safety_checker=None,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        if not self.config.progress_bar:
            self.model.set_progress_bar_config(disable=True)
        
        if config.checkpoint_path:
            checkpoint = load_file(config.checkpoint_path)
            self.model.unet.load_state_dict(checkpoint)
            
        self.model.to(config.device)
        
        # Enable attention slicing for memory efficiency
        self.model.enable_attention_slicing(config.attention_slicing_size)
        
        # Use DDIM scheduler
        self.model.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        
        # Move model to device after optimization setup
        self.model.enable_attention_slicing(config.attention_slicing_size)
        self.model.enable_vae_slicing()
        
        # Keep models in training mode for gradient computation
        # self.model.unet.train()
        # self.model.text_encoder.train()
        # self.model.vae.train()
        
        # Enable gradient checkpointing
        self.model.unet.enable_gradient_checkpointing()
        self.model.vae.enable_gradient_checkpointing()

    @staticmethod
    def generate_fixed_noise(
        seed: int,
        shape: Tuple[int, ...] = (1, 4, 64, 64),
        device: str = "cuda"
    ) -> torch.Tensor:
        """Generate deterministic initial noise."""
        torch.manual_seed(seed)
        return torch.randn(shape, device=device)

    def encode_prompt(
        self,
        prompt: str,
    ) -> torch.Tensor:
        """Encode the prompt to text embeddings."""
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            device=self.config.device,
            num_images_per_prompt=self.config.num_images_per_prompt,
            do_classifier_free_guidance=self.config.guidance_scale > 1
        )
        
        if self.config.guidance_scale > 1:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        return prompt_embeds.to(dtype=torch.float16)
    
    def generate(self, prompt: str, seed: int) -> torch.Tensor:
        """Generate image from prompt using Stable Diffusion."""
        seed_everything(seed)
        with torch.cuda.amp.autocast():
            # Setup
            self.model.scheduler.set_timesteps(self.config.num_inference_steps)
            timesteps = self.model.scheduler.timesteps
            
            # Generate initial noise
            latents = self.generate_fixed_noise(
                seed=seed,
                device=self.config.device
            ).to(dtype=torch.float16)
            
            # Encode prompt
            prompt_embeds = self.encode_prompt(prompt).type(torch.float16)
            
            # Main denoising loop
            for i, t in enumerate(timesteps):
                # Clear memory before each step
                torch.cuda.empty_cache()
                gc.collect()
                
                # Prepare latent input
                latent_model_input = torch.cat([latents] * 2) if self.config.guidance_scale > 1 else latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.model.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds
                    ).sample
                
                # Apply classifier-free guidance
                if self.config.guidance_scale > 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                
                # Compute previous noisy sample
                latents = self.model.scheduler.step(noise_pred, t, latents).prev_sample
            
            generator = None
            # Decode latents
            with torch.no_grad():
                image = self.model.vae.decode(latents / sd_inference.model.vae.config.scaling_factor, return_dict=False, generator=generator)[0].detach()
                
            return image

    def optimize(self, prompt_neutral: str, prompt_style: str, save_dir: str, seed: int, lambda_t_star: int) -> torch.Tensor:
        # Keep models in training mode for gradient computation
        self.model.unet.train()
        self.model.text_encoder.train()
        self.model.vae.train()
        
        seed_everything(seed)
        lambda_t_default_1 = self.config.lambda_t_default_1
        lambda_t_default_2 = self.config.lambda_t_default_2

        # Ensure lambda_t is created on CUDA from the start
        lambda_t = [lambda_t_default_1] * lambda_t_star + [
                lambda_t_default_2
            ] * (
                self.config.num_inference_steps - lambda_t_star
            )
            
        weighting_parameter = torch.tensor(lambda_t, requires_grad=True, device=self.config.device, dtype=torch.float32)
        print('Initial weighting parameter:', weighting_parameter)
        optimizer = optim.Adam([weighting_parameter], lr=0.05)
        scaler = torch.cuda.amp.GradScaler()

        num_images_per_prompt = 1
        
        # Pre-compute prompt embeddings and ensure they're on CUDA
        neutral_prompt_embeds, negative_neutral_prompt_embeds = self.model.encode_prompt(
            prompt=prompt_neutral, 
            device=self.config.device, 
            num_images_per_prompt=num_images_per_prompt, 
            do_classifier_free_guidance=True,
        )
        neutral_prompt_embeds = torch.cat([negative_neutral_prompt_embeds, neutral_prompt_embeds]).to(self.config.device)
            
        style_prompt_embeds, negative_style_prompt_embeds = self.model.encode_prompt(
            prompt=prompt_style, 
            device=self.config.device, 
            num_images_per_prompt=num_images_per_prompt, 
            do_classifier_free_guidance=True,
        )
        style_prompt_embeds = torch.cat([negative_style_prompt_embeds, style_prompt_embeds]).to(self.config.device)
        
        # Setup
        self.model.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.model.scheduler.timesteps
        
        # Generate initial noise
        latents = self.generate_fixed_noise(
            seed=seed,
            device=self.config.device
        ).to(dtype=torch.float16, device=self.config.device)
        
        # Generate original image once and ensure it's on CUDA
        with torch.no_grad():
            original_image = self.generate(prompt_neutral, seed)
        original_image = original_image.detach().to(device=self.config.device, dtype=torch.float32)
        print("Original image generated")
        torch.cuda.empty_cache()
        gc.collect()
        
        def unet_forward(latent_model_input, t, encoder_hidden_states):
            return self.model.unet(
                latent_model_input.to(self.config.device), 
                t.to(self.config.device), 
                encoder_hidden_states.to(self.config.device)
            ).sample
        os.makedirs(f"outputs/chexpert/seed{seed}_lambda_t_star{lambda_t_star}", exist_ok=True)
        for epoch in range(10):
            print(f"Epoch: {epoch}")
            optimizer.zero_grad()
            
            current_latents = latents.clone()
            
            # Main denoising loop
            for i, t in enumerate(timesteps):
                optimizing_weights = weighting_parameter[i].to(self.config.device)
                prompt_embeds = (optimizing_weights * neutral_prompt_embeds + 
                            (1 - optimizing_weights) * style_prompt_embeds).to(self.config.device)
                
                torch.cuda.empty_cache()
                gc.collect()
                
                # Prepare latent input
                latent_model_input = torch.cat([current_latents] * 2) if self.config.guidance_scale > 1 else current_latents
                latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                
                # Ensure all inputs to unet_forward are on CUDA
                with torch.cuda.amp.autocast():
                    noise_pred = checkpoint(
                        unet_forward,
                        latent_model_input.to(self.config.device),
                        t,
                        prompt_embeds.to(self.config.device)
                    )
                    
                    if self.config.guidance_scale > 1:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                
                current_latents = self.model.scheduler.step(
                    noise_pred.to(self.config.device), 
                    t.to(self.config.device), 
                    current_latents.to(self.config.device)
                ).prev_sample
            
            # Compute loss with mixed precision and ensure all tensors are on CUDA
            with torch.cuda.amp.autocast():
                generated_image = self.model.vae.decode(
                    current_latents.to(self.config.device) / self.model.vae.config.scaling_factor, 
                    return_dict=False, 
                    generator=None
                )[0]
                
                # Ensure inputs to loss functions are on CUDA with correct dtype
                generated_image = generated_image.to(self.config.device, dtype=torch.float32)
                
                loss1 = perceptual_loss(generated_image[0], original_image[0])
                loss2 = clip_loss(
                    original_image[0].to(self.config.device), 
                    generated_image[0].to(self.config.device), 
                    prompt_neutral, 
                    prompt_style
                )
                loss = (0.05 * loss1 + loss2).to(self.config.device)
                
            #propogate loss 
            optimizer.zero_grad()
            #debug_backward_devices(loss, self.model)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            print(f'Weighting parameter: {weighting_parameter}')
            print(f"Epoch {epoch} Loss: {loss.item()}")

        #save the generated image after each epoch
        normalized_original_image = (original_image[0] - original_image[0].min()) / (original_image[0].max() - original_image[0].min())
        normalized_generated_image = (generated_image[0] - generated_image[0].min()) / (generated_image[0].max() - generated_image[0].min())
        
        torchvision.utils.save_image(normalized_original_image, f"{save_dir}/neutral.png")
        torchvision.utils.save_image(normalized_generated_image, f"{save_dir}/style.png")    
        
        return generated_image, weighting_parameter

    def swap_embeddings(self, prompt_neutral: str, prompt_style: str, save_dir: str, seed: int, lambda_t_star: int):
        # seed_everything(seed)
        def swap_callback(pipe, step_index, timestep, callback_kwargs):
            num_images_per_prompt = 1

            if step_index == lambda_t_star:  # Change embedding after step 10
                prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
                    prompt=prompt_style, 
                    device=self.config.device, 
                    num_images_per_prompt=num_images_per_prompt, 
                    do_classifier_free_guidance=self.model.do_classifier_free_guidance,
                    )

                if self.model.do_classifier_free_guidance:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                            
                callback_kwargs['prompt_embeds'] = prompt_embeds
                callback_kwargs['negative_prompt_embeds'] = negative_prompt_embeds

            return callback_kwargs

        fixed_noise = self.generate_fixed_noise(seed=seed, device=self.config.device).type(torch.float16)
        
        original_latent = self.model(prompt=prompt_neutral, latents=fixed_noise, output_type='latent', progress_bar=False).images

        # Run diffusion with controlled conditioning
        latent = self.model(
            prompt=prompt_neutral,
            callback_on_step_end=swap_callback, 
            latents=fixed_noise,
            callback_on_step_end_tensor_inputs=['prompt_embeds', 'negative_prompt_embeds'],
            output_type='latent',
            num_inference_steps=self.config.num_inference_steps,
            progress_bar=False).images
        
        # Convert latents to images
        def latents_to_pil(latents, vae):
            latents = 1 / 0.18215 * latents  # Scale latents back
            image = vae.decode(latents).sample  # Decode latents to pixel space
            image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
            image = (image * 255).byte().cpu().numpy()  # Convert to [0,255] and NumPy
            image = image.transpose(0, 2, 3, 1)  # Rearrange (B, C, H, W) → (B, H, W, C)
            pil_images = [Image.fromarray(img) for img in image]  # Convert to PIL
            return pil_images
        
        image = latents_to_pil(latent, self.model.vae)[0]
        original_image = latents_to_pil(original_latent, self.model.vae)[0]
        
        # Save the images
        # print(f"Saving images to {save_dir}")
        original_image.save(f"{save_dir}/neutral.png")
        image.save(f"{save_dir}/style.png")

        # Save the latents
        torch.save(original_latent, f"{save_dir}/neutral.pt")
        torch.save(latent, f"{save_dir}/style.pt")
            
        return image

    
def run_inference_on_gpu(gpu_id, config, prompt_neutral, disease, style_prompt_dict, seeds, lambda_t_stars, total_tasks):
    # Set device before any CUDA operations
    torch.cuda.set_device(gpu_id)
    config.device = f'cuda:{gpu_id}'
    
    # Initialize model inside the process
    sd_inference = StableDiffusionInference(config)
    
    disease_name = "nevus" if "nevus" in disease else "melanoma"
    
    with tqdm(total=total_tasks, position=gpu_id, desc=f"GPU {gpu_id}") as pbar:
        for seed in seeds:
            for style, prompt_style in style_prompt_dict.items():
                for lambda_t_star in lambda_t_stars:
                    save_dir = f'/home/amarkr/dent/outputs/dent_outputs/isic2019_v2/{disease_name}/{style}/seed{seed}/lambda_t_star{lambda_t_star}'
                    os.makedirs(save_dir, exist_ok=True)
                    
                    try:
                        image = sd_inference.swap_embeddings(prompt_neutral, prompt_style, save_dir, seed, lambda_t_star)
                        torch.cuda.empty_cache()  # Clear CUDA cache after each iteration
                    except Exception as e:
                        print(f"Error on GPU {gpu_id} with seed {seed}, style {style}: {str(e)}")
                        continue
                    
                    pbar.update(1)

def main():
    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    hostname = socket.gethostname()
    checkpoint_path = {
        'kronos': None,
        'four-a100': '/home/amarkr/dent/saved_models/isic2019_finetune_sd1.5/checkpoint-300-0/model.safetensors',
        'inferentia': '/home/amarkr/dent/saved_models/isic2019_finetune_sd1.5/checkpoint-300-0/model.safetensors'
    }.get(hostname, None)
    
    print(f'Loading model from checkpoint: {checkpoint_path}')

    config = InferenceConfig(
        attention_slicing_size=4,
        num_inference_steps=50,
        checkpoint_path=checkpoint_path,
        progress_bar=False,
    )

    diseases = ["melanocytic nevus (NV)", "melanoma (MEL)"]
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    for disease in diseases:
        prompt_neutral = f"a dermoscopic image with {disease}"
        
        style_prompt_dict = {
            "ink": f"a dermoscopic image with {disease} showing ink",
            "gel_bubbles": f"a dermoscopic image with {disease} showing gel bubbles",
            "ruler": f"a dermoscopic image with {disease} with a ruler visible",
            "hairs": f"a dermoscopic image with {disease} with hairs visible on skin"
        }

        seeds = list(range(3000, 4000))
        lambda_t_stars = list(range(5, 46, 5))

        # Split work across GPUs
        seeds_per_gpu = [[] for _ in range(num_gpus)]
        for i, seed in enumerate(seeds):
            seeds_per_gpu[i % num_gpus].append(seed)

        total_tasks = len(style_prompt_dict) * len(seeds) * len(lambda_t_stars) // num_gpus

        # Create and start processes
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=run_inference_on_gpu,
                args=(
                    gpu_id,
                    config,
                    prompt_neutral,
                    disease,
                    style_prompt_dict,
                    seeds_per_gpu[gpu_id],
                    lambda_t_stars,
                    total_tasks
                )
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()