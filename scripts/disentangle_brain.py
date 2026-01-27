import os, socket
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
from pytorch_lightning import seed_everything
from dataclasses import dataclass
from typing import Optional, Tuple
import gc
import numpy as np
from torch import optim
import torchvision
import clip
from torch.utils.checkpoint import checkpoint

    
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
    seed: int = 70
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    attention_slicing_size: int = 4  # Added for memory optimization

class StableDiffusionInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        seed_everything(config.seed)
        
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
        self.model.unet.train()
        self.model.text_encoder.train()
        self.model.vae.train()
        
        # Enable gradient checkpointing
        self.model.unet.enable_gradient_checkpointing()
        try:
            self.model.text_encoder.enable_gradient_checkpointing()
        except AttributeError:
            print("Gradient checkpointing not available for text_encoder")
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
    
    def generate(self, prompt: str) -> torch.Tensor:
        """Generate image from prompt using Stable Diffusion."""
        with torch.cuda.amp.autocast():
            # Setup
            self.model.scheduler.set_timesteps(self.config.num_inference_steps)
            timesteps = self.model.scheduler.timesteps
            
            # Generate initial noise
            latents = self.generate_fixed_noise(
                seed=self.config.seed,
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

    def optimize(self, prompt_neutral: str, prompt_style: str, neutral: str, style: str) -> torch.Tensor:
        lambda_t_star = 10
        lambda_t_default_1 = 1.0
        lambda_t_default_2 = 0.0

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
            seed=self.config.seed,
            device=self.config.device
        ).to(dtype=torch.float16, device=self.config.device)
        
        # Generate original image once and ensure it's on CUDA
        with torch.no_grad():
            original_image = self.generate(prompt_neutral)
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
        os.makedirs(f"outputs/brats-ixi/seed{config.seed}_lambda_t_star{lambda_t_star}", exist_ok=True)
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
        #create path
        os.makedirs(f"outputs/brats-ixi/{neutral}-{style}/seed{config.seed}_lambda_t_star{lambda_t_star}", exist_ok=True)
        
        torchvision.utils.save_image(normalized_original_image, f"outputs/brats-ixi/{neutral}-{style}/seed{config.seed}_lambda_t_star{lambda_t_star}/neutral.png")
        torchvision.utils.save_image(normalized_generated_image, f"outputs/brats-ixi/{neutral}-{style}/seed{config.seed}_lambda_t_star{lambda_t_star}/style.png")    
        
        return generated_image, weighting_parameter
    
# Usage example
if __name__ == "__main__":
    if socket.gethostname() == 'kronos':
        checkpoint_path = None
    elif socket.gethostname() == 'four-a100':
        checkpoint_path = '/home/amarkr/dent/saved_models/finetuned_brain_256_sd1.5/checkpoint-100-0/model.safetensors'
    print('Loading model from the checkpoint:', checkpoint_path)   
    
    # try with few random seeds on different gpus
    for i in range(5):
    
    
        # Configure with memory optimizations
        config = InferenceConfig(
            attention_slicing_size=4,  # Adjust based on your GPU memory
            num_inference_steps=50,    # Reduce if still having memory issues
            checkpoint_path=checkpoint_path,
            seed=np.random.randint(0, 1000)
        )
        
        # Initialize and run
        sd_inference = StableDiffusionInference(config)
        prompt_neutral = "Healthy Brain MRI"
        prompt_style = "Brain MRI with large tumor"
        neutral = "healthy"
        style = "tumor"
        image, weighting_parameter = sd_inference.optimize(prompt_neutral, prompt_style, neutral, style)
        print(weighting_parameter)
