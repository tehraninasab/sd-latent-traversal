import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import multiprocessing
import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import argparse
import logging
from datetime import datetime
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data.sampler import WeightedRandomSampler

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(0, '/usr/local/data/zahrat/workshop/stable-diffusion-finetune')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
file_handler = logging.FileHandler(f'out/finetune_{timestamp}.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class HAM10KDataset(Dataset):
    def __init__(
        self,
        metadata,
        image_root_path,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.df = metadata
        
        print(self.df.groupby(['dx', 'split']).size())

        self.image_root_path = image_root_path
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        
        if center_crop:
            self.transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        self.set_image_paths()
        self.create_captions()
        
    def set_image_paths(self):
        image_paths = []

        for index, row in self.df.iterrows():
            image_id = row['image_id']
            
            base1_path = os.path.join(self.image_root_path, 'HAM10000_images_part_1')
            base2_path = os.path.join(self.image_root_path, 'HAM10000_images_part_2')
            
            image_path_1 = os.path.join(base1_path, f'{image_id}.jpg')
            image_path_2 = os.path.join(base2_path, f'{image_id}.jpg')

            if os.path.exists(image_path_1):
                image_paths.append(image_path_1)
            elif os.path.exists(image_path_2):
                image_paths.append(image_path_2)
            else:
                print(f'Image {image_id} not found in either path.')

            # Add the image paths to your DataFrame
        self.df['image_path'] = image_paths

    def create_captions(self):
        dx_dict = {
            "mel": "melanoma",
            "nv": "melanocytic nevi",
            "bkl": "benign keratosis",
            "bcc": "basal cell carcinoma",
            "akiec": "actinic keratosis",
            "vasc": "vascular lesions",
            "df": "dermatofibroma",
        }
        
        captions = []
        for _, row in self.df.iterrows():
            finding = dx_dict.get(row['dx'], None)
            
            caption = f"dermatoscopic image showing {finding}"
            captions.append(caption)
        
        self.df['caption'] = captions

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']  # Fixed image path handling
        caption = row['caption']

        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        encoding = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "image": image,
            "input_ids": encoding.input_ids[0].to(device),
            "attention_mask": encoding.attention_mask[0].to(device)
        }

def create_weighted_sampler(dataset):
    class_counts = dataset.df['dx'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[row['dx']] for _, row in dataset.df.iterrows()]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def train_one_epoch(
    accelerator,
    model,
    sampler,
    text_encoder,
    tokenizer,
    dataloader,
    optimizer,
    epoch,
    args
):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}")):
        with accelerator.accumulate(model):
            latents = model.module.encode_first_stage(batch["image"])  # Access the underlying model
            if isinstance(latents, DiagonalGaussianDistribution):
                latents = latents.sample()
            latents = latents * 0.18215  # scaling factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, model.module.num_timesteps, (bsz,), device=latents.device)
            noisy_latents = model.module.q_sample(x_start=latents, t=timesteps, noise=noise)

            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            noise_pred = model.module.apply_model(noisy_latents, timesteps, encoder_hidden_states)

            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.detach().item()

        if step % args.save_steps == 0 and epoch % args.save_epochs == 0:
            save_progress(accelerator, model, args.output_dir, epoch, step)

    return total_loss / len(dataloader)

def save_progress(accelerator, model, output_dir, epoch, step):
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f"checkpoint-{epoch}-{step}")
        accelerator.save_state(save_path)

def main(args):
    if __name__ == "__main__":
        multiprocessing.set_start_method('spawn', force=True)
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl')
        else:
            dist.init_process_group(backend='gloo')  # Added fallback for non-CUDA devices
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if dist.get_rank() == 0 else logging.WARN,
    )

    # Load models
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

    sampler = DDIMSampler(model)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    text_encoder = text_encoder.to(device)

    text_encoder.requires_grad_(False)

    # Setup training data
    df = pd.read_csv(args.metadata_path)
    df[df['split'] == 'train']
    train_dataset = HAM10KDataset(
        metadata=df,
        image_root_path=args.image_root_path,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_sampler = create_weighted_sampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        multiprocessing_context='spawn'
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # Training loop
    for epoch in range(args.num_train_epochs):
        train_loss = train_one_epoch(
            accelerator,
            model,
            sampler,
            text_encoder,
            tokenizer,
            train_dataloader,
            optimizer,
            epoch,
            args
        )
        
        if dist.get_rank() == 0:
            logger.info(f"Epoch {epoch}: Average loss = {train_loss}")
        
        if epoch == args.num_train_epochs - 1 and dist.get_rank() == 0:
            model.save_pretrained(args.output_dir)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on HAM10000 dataset")
    parser.add_argument("--model_name_or_path", type=str, default="CompVis/stable-diffusion-v-1-4-original")
    parser.add_argument("--metadata_path", type=str, default='data/ham10k_metadata.csv')
    parser.add_argument("--image_root_path", type=str, default='/usr/local/faststorage/datasets/zahrat/ham10000')
    parser.add_argument("--output_dir", type=str, default='/usr/local/data/zahrat/workshop/stable-diffusion-finetune/saved_models/ham10k-finetuned-sd-original-balancedsampler')
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--save_epochs", type=int, default=3)

    args = parser.parse_args()
    main(args)
