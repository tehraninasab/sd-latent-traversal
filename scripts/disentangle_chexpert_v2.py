import os, socket
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
from pytorch_lightning import seed_everything
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import gc
import numpy as np
from torch import optim
import torchvision
import clip
from torch.utils.checkpoint import checkpoint
from PIL import Image
import concurrent.futures
import argparse
from tqdm import tqdm
import time
import logging
import sys
from datetime import datetime

# Keep the original VGGPerceptualLoss and DCLIPLoss classes
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
    seed: int = 70
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    attention_slicing_size: int = 4  # Added for memory optimization
    lambda_t_star: int = 10  # For swap_embeddings method
    gpu_id: int = 0  # Added for multi-GPU support

class StableDiffusionInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        seed_everything(config.seed)
        
        # Choose specific GPU if specified
        if config.device == 'cuda' and torch.cuda.is_available():
            device_id = f"cuda:{config.gpu_id}" if config.gpu_id is not None else "cuda"
            self.config.device = device_id
        print("Initializing model on device: {self.config.device}")
        
        # Set memory optimization configs
        torch.cuda.empty_cache()
        gc.collect()
        
        # Force tensor cores usage for faster inference when available (A100, etc)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set CUDA benchmarking to optimize kernels for the specific hardware
        torch.backends.cudnn.benchmark = True
        
        # Initialize model with memory optimizations
        self.model = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            safety_checker=None,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            # Preload attention processors for faster model loading
            attn_implementation="flash_attention_2" if self._is_flash_attn_available() else "sdpa",
        )
        
        if config.checkpoint_path:
            checkpoint = load_file(config.checkpoint_path)
            self.model.unet.load_state_dict(checkpoint)
            
        self.model.to(self.config.device)
        
        # Enable maximum memory optimizations
        # Enable attention slicing for memory efficiency
        self.model.enable_attention_slicing(config.attention_slicing_size)
        
        # Use DDIM scheduler
        self.model.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        
        # Try to enable memory optimizations with xformers if available
        self._enable_xformers_memory_efficient_attention()
        
        # Model optimizations for inference
        self.model.enable_vae_slicing()
        self.model.enable_vae_tiling()
        
        # Keep models in training mode for gradient computation
        self.model.unet.train()
        self.model.text_encoder.train()
        self.model.vae.train()
        
        # Enable gradient checkpointing for memory efficiency during backprop
        self.model.unet.enable_gradient_checkpointing()
        try:
            self.model.text_encoder.enable_gradient_checkpointing()
        except AttributeError:
            print("Gradient checkpointing not available for text_encoder")
        self.model.vae.enable_gradient_checkpointing()
        
    def _is_flash_attn_available(self):
        """Check if flash attention is available"""
        try:
            from diffusers.utils import is_flash_attn_2_available
            return is_flash_attn_2_available()
        except ImportError:
            return False
            
    def _enable_xformers_memory_efficient_attention(self):
        """Enable xformers memory efficient attention if available"""
        try:
            self.model.enable_xformers_memory_efficient_attention()
            print("Successfully enabled xformers memory efficient attention")
        except (ImportError, AttributeError):
            print("xformers is not available, using default attention mechanism")

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
                image = self.model.vae.decode(latents / self.model.vae.config.scaling_factor, return_dict=False, generator=generator)[0].detach()
                
            return image

    def swap_embeddings(self, prompt_neutral: str, prompt_style: str, file_prefix: str, save_dir: str):
        def swap_callback(pipe, step_index, timestep, callback_kwargs):
            num_images_per_prompt = 1

            if step_index == self.config.lambda_t_star:  # Change embedding at lambda_t_star step
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

        # Generate fixed noise with proper memory handling
        with torch.cuda.amp.autocast():
            fixed_noise = self.generate_fixed_noise(seed=self.config.seed, device=self.config.device).type(torch.float16)

        # Pre-compute latents to maximize GPU utilization
        with torch.cuda.amp.autocast(), torch.no_grad():
            # Generate original image
            original_latent = self.model(
                prompt=prompt_neutral, 
                latents=fixed_noise, 
                output_type='latent',
                num_inference_steps=self.config.num_inference_steps,
            ).images

            # Run diffusion with controlled conditioning - optimize for throughput
            latent = self.model(
                prompt=prompt_neutral,
                callback_on_step_end=swap_callback, 
                latents=fixed_noise,
                callback_on_step_end_tensor_inputs=['prompt_embeds', 'negative_prompt_embeds'],
                output_type='latent',
                num_inference_steps=self.config.num_inference_steps,
            ).images
        
        # Optimize latent to image conversion
        def latents_to_pil(latents, vae):
            # Use non-blocking tensor operations where possible
            with torch.cuda.amp.autocast(), torch.no_grad():
                # Scale latents back with optimal precision
                scaled_latents = 1 / 0.18215 * latents
                
                # Process in chunks if needed to avoid OOM
                if latents.shape[0] > 4:
                    chunks = []
                    for i in range(0, latents.shape[0], 4):
                        chunk = vae.decode(scaled_latents[i:i+4]).sample
                        chunks.append(chunk)
                    image = torch.cat(chunks, dim=0)
                else:
                    image = vae.decode(scaled_latents).sample
                
                # Normalize and convert efficiently
                image = (image / 2 + 0.5).clamp(0, 1)
                
                # Move to CPU asynchronously if possible
                image = image.cpu()
                
                # Convert to numpy and rearrange after leaving GPU
                image = (image * 255).byte().numpy()
                image = image.transpose(0, 2, 3, 1)
                
                # Convert to PIL
                pil_images = [Image.fromarray(img) for img in image]
            return pil_images
        
        # Create directory early to avoid race conditions
        os.makedirs(save_dir, exist_ok=True)
        
        # Process and save images with error handling
        try:
            image = latents_to_pil(latent, self.model.vae)[0]
            original_image = latents_to_pil(original_latent, self.model.vae)[0]
            
            # Save images, ensuring directory exists
            original_image.save(f"{save_dir}/{file_prefix}_neutral.png")
            image.save(f"{save_dir}/{file_prefix}_style.png")
            
            # Clean GPU memory after processing both images
            del latent, original_latent
            torch.cuda.empty_cache()
                
            return image
            
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            # Clean memory even on failure
            try:
                del latent, original_latent
            except:
                pass
            torch.cuda.empty_cache()
            raise

# Function to process a single task (for parallelization)
def process_task(task_config):
    try:
        # Unpack the task configuration
        prompt_neutral = task_config['prompt_neutral']
        style_name = task_config['style_name'] 
        style_prompt = task_config['style_prompt']
        seed = task_config['seed']
        lambda_t_star = task_config['lambda_t_star']
        base_dir = task_config['base_dir']
        gpu_id = task_config['gpu_id']
        checkpoint_path = task_config['checkpoint_path']
        
        # Create directory path
        style_dir = f"{base_dir}/{style_name}"
        os.makedirs(style_dir, exist_ok=True)
        
        # Create filename prefix with meaningful parameters
        file_prefix = f"seed{seed}_lambda_t_star{lambda_t_star}_gpu{gpu_id}"
        
        # Initialize SD inference with specific GPU
        config = InferenceConfig(
            attention_slicing_size=4,
            num_inference_steps=50,
            checkpoint_path=checkpoint_path,
            seed=seed,
            lambda_t_star=lambda_t_star,
            gpu_id=gpu_id
        )
        
        # Create a new inference instance for this task
        print(f"[Worker GPU:{gpu_id}] Processing {style_name} with seed={seed}, lambda_t_star={lambda_t_star}")
        sd_inference = StableDiffusionInference(config)
        
        # Process the task
        image = sd_inference.swap_embeddings(
            prompt_neutral=prompt_neutral, 
            prompt_style=style_prompt, 
            file_prefix=file_prefix,
            save_dir=style_dir
        )
        
        # Clean up GPU memory
        del sd_inference
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"[Worker GPU:{gpu_id}] Completed {style_name} with seed={seed}, lambda_t_star={lambda_t_star}")
        
        return {
            "status": "success",
            "style_name": style_name,
            "seed": seed,
            "lambda_t_star": lambda_t_star,
            "gpu_id": gpu_id
        }
        
    except Exception as e:
        print(f"Error in worker GPU:{gpu_id}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "style_name": style_name,
            "seed": seed, 
            "lambda_t_star": lambda_t_star,
            "gpu_id": gpu_id
        }

def create_task_batches(styles, lambda_t_stars, seeds, num_gpus):
    """Create task batches balanced across available GPUs"""
    
    all_tasks = []
    for style_name, style_prompt in styles.items():
        for seed in seeds:
            for lambda_t_star in lambda_t_stars:
                all_tasks.append({
                    'style_name': style_name,
                    'style_prompt': style_prompt,
                    'seed': seed,
                    'lambda_t_star': lambda_t_star
                })
    
    # Distribute tasks evenly across GPUs
    gpu_task_batches = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        gpu_id = i % num_gpus
        gpu_task_batches[gpu_id].append(task)
    
    return gpu_task_batches

def run_parallel_processing(gpu_task_batches, base_dir, checkpoint_path, prompt_neutral):
    """Run tasks in parallel across multiple processes, one per GPU, with a shared task queue for better load balancing"""
    main_logger = logging.getLogger('stable_diffusion.main')
    
    # Create a shared task queue and result queue
    task_queue = torch.multiprocessing.Queue()
    result_queue = torch.multiprocessing.Queue()
    
    # Initialize model on each GPU once to avoid repeated initialization
    num_gpus = len(gpu_task_batches)
    
    # Flatten all tasks and put them in the queue
    total_tasks = 0
    task_stats = {}  # For collecting statistics
    
    main_logger.info("Preparing task queue...")
    for batch in gpu_task_batches:
        for task in batch:
            task['base_dir'] = base_dir
            task['checkpoint_path'] = checkpoint_path
            task['prompt_neutral'] = prompt_neutral
            task_queue.put(task)
            
            # Track task distribution for analytics
            style_name = task['style_name']
            if style_name not in task_stats:
                task_stats[style_name] = 0
            task_stats[style_name] += 1
            
            total_tasks += 1
    
    # Log task distribution
    main_logger.info(f"Task distribution by style:")
    for style, count in task_stats.items():
        main_logger.info(f"  {style}: {count} tasks ({count/total_tasks*100:.1f}%)")
    
    # Add termination signals
    for _ in range(num_gpus):
        task_queue.put(None)  # Sentinel to signal worker to terminate
    
    main_logger.info(f"Starting {num_gpus} worker processes")
    
    # Start GPU workers
    processes = []
    for gpu_id in range(num_gpus):
        process = torch.multiprocessing.Process(
            target=dynamic_gpu_worker, 
            args=(gpu_id, task_queue, result_queue, total_tasks)
        )
        processes.append(process)
        process.start()
        main_logger.debug(f"Started worker process for GPU {gpu_id}, PID: {process.pid}")
    
    # Prepare for statistics collection
    completed_tasks = 0
    failed_tasks = 0
    style_completion_times = {}
    gpu_completion_times = {i: [] for i in range(num_gpus)}
    
    # Monitor progress
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        while completed_tasks + failed_tasks < total_tasks:
            if not result_queue.empty():
                result = result_queue.get()
                if result['status'] == 'success':
                    completed_tasks += 1
                    
                    # Collect performance statistics
                    if 'duration' in result:
                        style = result['style_name']
                        if style not in style_completion_times:
                            style_completion_times[style] = []
                        style_completion_times[style].append(result['duration'])
                        gpu_completion_times[result['gpu_id']].append(result['duration'])
                        
                    main_logger.debug(
                        f"Task completed: style={result['style_name']}, "
                        f"seed={result['seed']}, lambda={result['lambda_t_star']}, "
                        f"GPU={result['gpu_id']}"
                    )
                else:
                    failed_tasks += 1
                    main_logger.warning(
                        f"Task failed: style={result['style_name']}, "
                        f"seed={result['seed']}, lambda={result['lambda_t_star']}, "
                        f"error={result.get('error', 'Unknown error')}"
                    )
                pbar.update(1)
                
                # Periodically log progress
                if (completed_tasks + failed_tasks) % 10 == 0 or (completed_tasks + failed_tasks) == total_tasks:
                    progress_pct = (completed_tasks + failed_tasks) / total_tasks * 100
                    main_logger.info(
                        f"Progress: {completed_tasks + failed_tasks}/{total_tasks} tasks "
                        f"({progress_pct:.1f}%) - {completed_tasks} successful, {failed_tasks} failed"
                    )
            else:
                time.sleep(0.1)  # Prevent CPU spinning
    
    # Wait for all processes to complete
    main_logger.info("All tasks completed, waiting for worker processes to terminate...")
    for i, process in enumerate(processes):
        process.join()
        main_logger.debug(f"Worker process for GPU {i} terminated, exit code: {process.exitcode}")
    
    # Log performance statistics
    main_logger.info(f"Processing complete. {completed_tasks} tasks succeeded, {failed_tasks} tasks failed.")
    
    if style_completion_times:
        main_logger.info("Performance statistics by style:")
        for style, times in style_completion_times.items():
            avg_time = sum(times) / len(times)
            main_logger.info(f"  {style}: {len(times)} tasks, avg time: {avg_time:.2f}s")
            
    if gpu_completion_times:
        main_logger.info("Performance statistics by GPU:")
        for gpu_id, times in gpu_completion_times.items():
            if times:
                avg_time = sum(times) / len(times)
                main_logger.info(f"  GPU {gpu_id}: {len(times)} tasks, avg time: {avg_time:.2f}s")
    
    return {
        "completed": completed_tasks,
        "failed": failed_tasks,
        "style_stats": style_completion_times,
        "gpu_stats": gpu_completion_times
    }

def dynamic_gpu_worker(gpu_id, task_queue, result_queue, total_tasks):
    """Worker that dynamically pulls tasks from queue for better GPU utilization"""
    worker_logger = logging.getLogger(f'stable_diffusion.worker.gpu{gpu_id}')
    worker_logger.info(f"Starting worker on GPU {gpu_id}")
    
    # Set the CUDA device for this process
    torch.cuda.set_device(gpu_id)
    
    # Pre-initialize model for this GPU
    config = InferenceConfig(
        attention_slicing_size=4,
        num_inference_steps=50,
        gpu_id=gpu_id
    )
    
    try:
        # Initialize the model once per worker
        worker_logger.debug("Initializing model")
        sd_inference = StableDiffusionInference(config)
        worker_logger.debug("Model initialized successfully")
        
        tasks_completed = 0
        
        # Process tasks from the queue
        while True:
            # Get a task from the queue
            task = task_queue.get()
            
            # Check for termination signal
            if task is None:
                worker_logger.info(f"Received termination signal after completing {tasks_completed} tasks")
                break
            
            # Configure this task for the current GPU
            task['gpu_id'] = gpu_id
            checkpoint_path = task['checkpoint_path']
            
            worker_logger.debug(
                f"Processing task: style={task['style_name']}, "
                f"seed={task['seed']}, lambda_t_star={task['lambda_t_star']}"
            )
            
            # Update model configuration for this specific task
            sd_inference.config.seed = task['seed']
            sd_inference.config.lambda_t_star = task['lambda_t_star']
            
            # Process the specific task
            try:
                style_dir = f"{task['base_dir']}/{task['style_name']}"
                os.makedirs(style_dir, exist_ok=True)
                
                file_prefix = f"seed{task['seed']}_lambda_t_star{task['lambda_t_star']}_gpu{gpu_id}"
                
                # Process the image
                start_task_time = time.time()
                sd_inference.swap_embeddings(
                    prompt_neutral=task['prompt_neutral'],
                    prompt_style=task['style_prompt'],
                    file_prefix=file_prefix,
                    save_dir=style_dir
                )
                task_duration = time.time() - start_task_time
                
                # Report success
                tasks_completed += 1
                worker_logger.info(
                    f"Completed task: style={task['style_name']}, "
                    f"seed={task['seed']}, lambda_t_star={task['lambda_t_star']}, "
                    f"duration={task_duration:.2f}s"
                )
                
                result_queue.put({
                    "status": "success",
                    "style_name": task['style_name'],
                    "seed": task['seed'],
                    "lambda_t_star": task['lambda_t_star'],
                    "gpu_id": gpu_id,
                    "duration": task_duration
                })
                
            except Exception as e:
                # Report failure but continue processing
                worker_logger.error(f"Error processing task: {str(e)}")
                worker_logger.debug(f"Task details: {task}", exc_info=True)
                
                result_queue.put({
                    "status": "error",
                    "error": str(e),
                    "style_name": task['style_name'],
                    "seed": task['seed'],
                    "lambda_t_star": task['lambda_t_star'],
                    "gpu_id": gpu_id
                })
            
            # Clean up between tasks
            torch.cuda.empty_cache()
            gc.collect()
            worker_logger.debug(f"Memory cleaned after task, available: {torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)}")
    
    except Exception as e:
        worker_logger.critical(f"Fatal error in worker: {str(e)}", exc_info=True)
    
    finally:
        # Final cleanup
        try:
            del sd_inference
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        worker_logger.info(f"Worker shutting down after completing {tasks_completed} tasks")

def setup_logger(log_level, output_dir):
    """Set up logger with file and console handlers"""
    # Create logger
    logger = logging.getLogger('stable_diffusion')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join('out'), exist_ok=True)
    log_file = os.path.join('out', f'image_generation_chexpert_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    # Enable better multiprocessing support
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run Stable Diffusion with multiple workers")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), 
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--output_dir", type=str, default="/cim/zahrat/workshop/dent/outputs/chexpert/multiple_styles",
                        help="Base directory for outputs")
    parser.add_argument("--start_seed", type=int, default=1500, 
                        help="Starting seed value")
    parser.add_argument("--num_seeds", type=int, default=3000, 
                        help="Number of seeds to process")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of seeds to process in a single inference batch (experimental)")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="Enable additional memory optimizations (may reduce speed)")
    parser.add_argument("--monitor_gpu", action="store_true",
                        help="Monitor GPU memory usage during processing")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set logging level")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level, args.output_dir)
    
    # GPU monitoring if requested
    if args.monitor_gpu:
        import threading
        def monitor_gpus():
            while True:
                gpu_info = ["GPU Memory Usage:"]
                for i in range(torch.cuda.device_count()):
                    mem_used = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    gpu_info.append(f"GPU {i}: {mem_used:.2f}GB / {mem_total:.2f}GB ({mem_used/mem_total*100:.1f}%)")
                logger.debug('\n'.join(gpu_info))
                time.sleep(10)
        monitor_thread = threading.Thread(target=monitor_gpus, daemon=True)
        monitor_thread.start()
    
    # Determine checkpoint path based on hostname
    if socket.gethostname() == 'progress':
        checkpoint_path = '/cim/zahrat/workshop/dent/saved_models/finetuned_chexpert_19Feb/checkpoint-60-0/model.safetensors'
    if socket.gethostname() == 'kronos':
        checkpoint_path = '/usr/local/data/zahrat/workshop/dent/saved_models/finetuned_chexpert-sd1.5/checkpoint-9-2500/model.safetensors'
    else:
        checkpoint_path = None

    logger.info(f'Loading model from the checkpoint: {checkpoint_path}')
    
    # Define the neutral prompt
    prompt_neutral = "Normal chest X-ray with no significant findings"
    
    # Define different style prompts
    styles = {
        "support_devices": "Chest X-ray showing Support Devices",
        "pleural_effusion": "Chest X-ray showing Pleural Effusion",
        "pleural_effusion-support_devices": "Chest X-ray showing Pleural Effusion, Support Devices",
    }
    
    # Define lambda_t_star values to try (when in diffusion process to switch style)
    lambda_t_stars = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    
    # Set up seed range
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    
    # Number of available GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    # Log GPU information
    logger.info(f"Using {num_gpus} GPUs out of {torch.cuda.device_count()} available")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Create base directorIt y
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    
    # Distribute tasks across GPUs
    gpu_task_batches = create_task_batches(styles, lambda_t_stars, seeds, num_gpus)
    
    # Calculate total tasks
    total_tasks = sum(len(batch) for batch in gpu_task_batches)
    logger.info(f"Total tasks to process: {total_tasks}")
    
    # Log task distribution summary
    for gpu_id, tasks in enumerate(gpu_task_batches):
        logger.info(f"GPU {gpu_id}: {len(tasks)} tasks assigned")
    
    # Start parallel processing with dynamic queue
    logger.info(f"Starting parallel processing with {num_gpus} workers")
    start_time = time.time()
    run_parallel_processing(gpu_task_batches, base_dir, checkpoint_path, prompt_neutral)
    
    # Log execution summary
    total_time = time.time() - start_time
    logger.info(f"All processing complete!")
    logger.info(f"Total execution time: {total_time:.1f} seconds")
    logger.info(f"Average time per task: {total_time/total_tasks:.1f} seconds")
    logger.info(f"Output saved to: {base_dir}")
