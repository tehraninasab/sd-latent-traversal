import os, socket
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
import clip
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import logging
from datetime import datetime


# Setup logging configuration
def setup_logger(save_dir: str, name: str = "stable_diffusion") -> logging.Logger:
    """Configure and return a logger instance."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H')
    file_handler = logging.FileHandler(
        os.path.join(f'out/imgGen_isic2019_{timestamp}.log')
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

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

@dataclass
class InferenceConfig:
    checkpoint_path: Optional[str] = None
    device: str = 'cuda'
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    attention_slicing_size: int = 4
    lambda_t_default_1: float = 1.0
    lambda_t_default_2: float = 0.0
    progress_bar: bool = False
    log_dir: str = 'logs'

class StableDiffusionInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = setup_logger(config.log_dir)
        
        self.logger.info("Initializing StableDiffusionInference")
        self.logger.info(f"Config: {config}")
        
        # Set memory optimization configs
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Initialize model with memory optimizations
            self.logger.info("Loading Stable Diffusion model...")
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
                self.logger.info(f"Loading checkpoint from {config.checkpoint_path}")
                checkpoint = load_file(config.checkpoint_path)
                self.model.unet.load_state_dict(checkpoint)
                
            self.model.to(config.device)
            self.logger.info(f"Model successfully loaded and moved to {config.device}")
            
            # Enable optimizations
            self.model.enable_attention_slicing(config.attention_slicing_size)
            self.model.scheduler = DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler"
            )
            self.model.enable_vae_slicing()
            
            # Keep models in training mode for gradient computation
            self.model.unet.train()
            self.model.text_encoder.train()
            self.model.vae.train()
            
            # Enable gradient checkpointing
            self.model.unet.enable_gradient_checkpointing()
            self.model.vae.enable_gradient_checkpointing()
        
            # Initialize loss functions
            self.logger.info("Initializing loss functions...")
            self.perceptual_loss = VGGPerceptualLoss().to(config.device)
            self.clip_loss = DCLIPLoss().to(config.device)
            
            self.logger.info("Model initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
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
        self.logger.info(f"Starting image generation with prompt: '{prompt}' and seed: {seed}")
        
        try:
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
                self.logger.info(f"Starting denoising loop with {len(timesteps)} steps")
                for i, t in enumerate(timesteps):
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    latent_model_input = torch.cat([latents] * 2) if self.config.guidance_scale > 1 else latents
                    latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
                    
                    with torch.no_grad():
                        noise_pred = self.model.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds
                        ).sample
                    
                    if self.config.guidance_scale > 1:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    
                    latents = self.model.scheduler.step(noise_pred, t, latents).prev_sample
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"Completed denoising step {i + 1}/{len(timesteps)}")
                
                # Decode latents
                self.logger.info("Decoding latents to image")
                with torch.no_grad():
                    image = self.model.vae.decode(latents / self.model.vae.config.scaling_factor, return_dict=False)[0].detach()
                
                self.logger.info("Image generation completed successfully")
                return image
                
        except Exception as e:
            self.logger.error(f"Error during image generation: {str(e)}", exc_info=True)
            raise

    def swap_embeddings(self, prompt_neutral: str, style: str, prompt_style: str, save_dir: str, seed: int, lambda_t_star: int):
        self.logger.info(f"Starting embedding swap with parameters:")
        self.logger.info(f"Neutral prompt: '{prompt_neutral}'")
        self.logger.info(f"Style: {style}")
        self.logger.info(f"Style prompt: '{prompt_style}'")
        self.logger.info(f"Lambda t*: {lambda_t_star}")
        
        try:
            def swap_callback(pipe, step_index, timestep, callback_kwargs):
                if step_index == lambda_t_star:
                    self.logger.info(f"Swapping embeddings at step {step_index}")
                    prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
                        prompt=prompt_style, 
                        device=self.config.device, 
                        num_images_per_prompt=self.config.num_images_per_prompt, 
                        do_classifier_free_guidance=self.model.do_classifier_free_guidance,
                    )

                    if self.model.do_classifier_free_guidance:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                                
                    callback_kwargs['prompt_embeds'] = prompt_embeds
                    callback_kwargs['negative_prompt_embeds'] = negative_prompt_embeds

                return callback_kwargs

            seed_everything(seed)
            fixed_noise = self.generate_fixed_noise(
                seed=seed,
                device=self.config.device
            ).to(dtype=torch.float16)
            
            self.logger.info("Generating original image with neutral prompt")
            original_latent = self.model(
                prompt=prompt_neutral,
                latents=fixed_noise,
                output_type='latent',
                progress_bar=False
            ).images

            self.logger.info("Running diffusion with controlled conditioning")
            latent = self.model(
                prompt=prompt_neutral,
                callback_on_step_end=swap_callback, 
                latents=fixed_noise,
                callback_on_step_end_tensor_inputs=['prompt_embeds', 'negative_prompt_embeds'],
                output_type='latent',
                num_inference_steps=self.config.num_inference_steps,
                progress_bar=False
            ).images
            
            # Convert and save images
            self.logger.info("Converting latents to images and saving")
            image = self._latents_to_pil(latent)[0]
            original_image = self._latents_to_pil(original_latent)[0]
            
            os.makedirs(save_dir, exist_ok=True)
            file_prefix = f"{style}_seed{seed}_lambdaTstar{lambda_t_star}"
            
            if lambda_t_star == 5:
                save_path = f"{save_dir}/{file_prefix}_neutral.png"
                self.logger.info(f"Saving neutral image to {save_path}")
                original_image.save(save_path)
            
            save_path = f"{save_dir}/{file_prefix}_style.png"
            self.logger.info(f"Saving styled image to {save_path}")
            image.save(save_path)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error during embedding swap: {str(e)}", exc_info=True)
            raise

    def _latents_to_pil(self, latents):
        """Helper method to convert latents to PIL images."""
        latents = 1 / 0.18215 * latents
        image = self.model.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).byte().cpu().numpy()
        image = image.transpose(0, 2, 3, 1)
        return [Image.fromarray(img) for img in image]

def run_inference_on_gpu(gpu_id, config, prompt_neutral, style_prompt_dict, seeds, lambda_t_stars, total_tasks, save_dir):
    logger = setup_logger(config.log_dir, f"gpu_{gpu_id}")
    logger.info(f"Initializing GPU {gpu_id} for inference")
    
    try:
        torch.cuda.set_device(gpu_id)
        config.device = f'cuda:{gpu_id}'
        sd_inference = StableDiffusionInference(config)
        
        with tqdm(total=total_tasks, position=gpu_id, desc=f"GPU {gpu_id}") as pbar:
            for seed in seeds:
                for style, prompt_style in style_prompt_dict.items():
                    for lambda_t_star in lambda_t_stars:
                        logger.info(f"Processing on GPU {gpu_id}: seed={seed}, style={style}, lambda_t_star={lambda_t_star}")
                        
                        os.makedirs(save_dir, exist_ok=True)
                        sd_inference.swap_embeddings(
                            prompt_neutral,
                            style, 
                            prompt_style, 
                            save_dir, 
                            seed, 
                            lambda_t_star
                        )
                        pbar.update(1)
                        
        logger.info(f"GPU {gpu_id} completed all tasks successfully")
        
    except Exception as e:
        logger.error(f"Error on GPU {gpu_id}: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # Setup main logger
    main_logger = setup_logger('logs', 'main')
    main_logger.info("Starting main process")

    # Determine checkpoint path based on hostname
    hostname = socket.gethostname()
    main_logger.info(f"Running on host: {hostname}")
    
    if hostname == 'kronos':
        checkpoint_path = None
    elif hostname == 'four-a100':
        checkpoint_path = '/home/amarkr/dent/saved_models/isic2019_finetune_sd1.5/checkpoint-300-0/model.safetensors'
    else:
        checkpoint_path = None
    main_logger.info(f'Loading model from checkpoint: {checkpoint_path}')

    # Configure with memory optimizations
    config = InferenceConfig(
        attention_slicing_size=4,
        num_inference_steps=50,
        checkpoint_path=checkpoint_path,
        progress_bar=False,
        log_dir='logs'
    )
    
    diseases = ["melanoma (MEL)", "melanocytic nevus (NV)"]
    for disease in diseases:
        main_logger.info(f"Processing disease: {disease}")
        prompt_neutral = f"a dermoscopic image with {disease}"
        
        if "nevus" in disease:
            save_dir = f'/home/jupyter/dent_outputs/isic2019/nevus/'
        else:
            save_dir = f'/home/jupyter/dent_outputs/isic2019/melanoma/'
        
        main_logger.info(f"Output directory: {save_dir}")
        
        style_prompt_dict = {
            "ink": f"a dermoscopic image with {disease} showing ink",
            "gel_bubbles": f"a dermoscopic image with {disease} showing gel bubbles",
            "ruler": f"a dermoscopic image with {disease} with a ruler visible",
            "dark_corners": f"a dermoscopic image with {disease} showing dark corners",
            "hairs": f"a dermoscopic image with {disease} with hairs visible on skin"
        }

        seeds = list(range(100, 2100))
        lambda_t_stars = [5, 10, 20, 30, 40]

        num_gpus = torch.cuda.device_count()
        main_logger.info(f"Number of available GPUs: {num_gpus}")

        # Split seeds for each GPU using remainder operator
        seeds_split = [[] for _ in range(num_gpus)]
        for i, seed in enumerate(seeds):
            seeds_split[i % num_gpus].append(seed)

        main_logger.info(f"Split {len(seeds)} seeds across {num_gpus} GPUs")
        for gpu_id, gpu_seeds in enumerate(seeds_split):
            main_logger.info(f"GPU {gpu_id} assigned {len(gpu_seeds)} seeds: {gpu_seeds}")

        # Calculate total tasks for progress bar
        total_tasks = len(style_prompt_dict) * len(seeds_split[0]) * len(lambda_t_stars)
        main_logger.info(f"Total tasks per GPU: {total_tasks}")

        try:
            # Create a pool of processes, one for each GPU
            main_logger.info("Starting multiprocessing pool")
            with Pool(num_gpus) as pool:
                pool.starmap(
                    run_inference_on_gpu,
                    [(gpu_id, config, prompt_neutral, style_prompt_dict, seeds_split[gpu_id], 
                      lambda_t_stars, total_tasks, save_dir) 
                     for gpu_id in range(num_gpus)]
                )
            main_logger.info("All GPU processes completed successfully")
            
        except Exception as e:
            main_logger.error(f"Error in main process: {str(e)}", exc_info=True)
            raise
        
        main_logger.info(f"Completed processing for disease: {disease}")
    
    main_logger.info("All processing completed successfully")