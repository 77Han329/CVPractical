#%pip install torchvision -q
#%pip install pytorch-lightning -q
#%pip install torchmetrics[image] -q


import os
import csv
import torch
import scipy.io
import numpy as np
import torch.nn.functional as F
import argparse
import random
from copy import deepcopy

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from SiT.train import center_crop_arr
from SiT.models import SiT_models
from SiT.download import find_model
from validation_loss.image_ldm_main.ldm.trainer import TrainerModuleLatentFlow
from validation_loss.image_ldm_main.ldm.flow import Flow
from glob import glob
from validation_loss.image_ldm_main.ldm.metrics import ImageMetricTracker
from torchmetrics.aggregation import CatMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageNetValDatasetWithLabels(Dataset):
    def __init__(self, image_dir, label_txt_path, meta_mat_path, synset_words_path, transform=None, max_images=1000):
        all_paths = sorted(glob(os.path.join(image_dir, "*")))
        print(len(all_paths), "images found in", image_dir)
        random.seed(42)  # Fixed seed ensures reproducibility
        random.shuffle(all_paths)
        self.image_paths = all_paths[:max_images]

        self.labels = self._load_labels(label_txt_path, meta_mat_path, synset_words_path)[:max_images]

        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def _load_labels(self, label_txt_path, meta_mat_path, synset_words_path):
        # Load synset mapping from .mat
        meta = scipy.io.loadmat(meta_mat_path)
        original_idx_to_synset = {
            int(entry[0][0][0]): entry[1][0]
            for entry in meta["synsets"].squeeze()[:1000]
        }

        # Load synset -> 0-999 index from synset_words.txt
        with open(synset_words_path, "r") as f:
            lines = f.readlines()
        synset_to_idx = {line.split(" ")[0]: i for i, line in enumerate(lines)}

        # Mapping function: ILSVRC ID → synset → index 0–999
        def to_idx(original_id):
            synset = original_idx_to_synset[int(original_id)]
            return synset_to_idx[synset]

        # Load raw labels
        with open(label_txt_path, "r") as f:
            raw_ids = [int(x.strip()) for x in f.readlines()]
        return np.array([to_idx(rid) for rid in raw_ids])

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_paths)

    
def center_crop_256(img):
    return center_crop_arr(img, 256)


def prepare_dataset(batch_size=64, num_samples=1000):
    val_dataset = ImageNetValDatasetWithLabels(
        image_dir="/home/coder/ILSVRC2012_img_val",
        label_txt_path="validation_loss/imagenet_data/ILSVRC2012_validation_ground_truth.txt",
        meta_mat_path="validation_loss/imagenet_data/meta.mat",
        synset_words_path="validation_loss/imagenet_data/synset_words.txt",
        max_images=num_samples,
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False,
    )

    print(f"Validation set prepared with {len(val_loader.dataset)} images.")
    return val_loader


def prepare_model():
    img_features = 256
    model = SiT_models['SiT-XL/2'](
        input_size=img_features//8,
        num_classes=1000
    )

    ckpt_path = "SiT-XL-2-256x256.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  

    cfg_path = "validation_loss/image_ldm_main/configs/model/sit-xl-2.yaml"
    cfg = OmegaConf.load(cfg_path)
    print(f"Loaded configuration from {cfg_path}, starting with TrainerModuleLatentFlow...")

    module = TrainerModuleLatentFlow(
        model=model,
        flow_cfg=cfg
    )   
    print("TrainerModuleLatentFlow initialized with the model and configuration.")

    module.eval()
    module.to(device)
    print("Model prepared and loaded with pre-trained weights.")

    return module


def compute_loss(module, val_loader, save_path="validation_loss/output/val_loss.csv"):
    print("Running validation...")
    
    # 1. Create Flow instance (match your training configuration)
    flow = Flow(schedule="linear")  # Or whatever config you used for training
    
    # 2. Get the model (SiT) - use EMA if available
    model = module.ema_model if hasattr(module, 'ema_model') else module.model
    model.eval()
    
    all_losses = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
        ims = batch["image"].to(module.device)
        labels = batch.get("label", None)
        if labels is not None:
            labels = labels.to(module.device)
        
        # 3. Encode to latents and resize for SiT
        with torch.no_grad():
            latent = module.encode(ims)  # [B,C,H,W]

            # Pad to 4 channels if needed
            if latent.size(1) == 3:
                latent = F.pad(latent, (0,0,0,0,0,1))  # Adds 1 channel

            # SiT expects latents of size 32x32
            if latent.shape[-2:] != (32, 32):
                latent = F.interpolate(latent, size=(32, 32), mode='bilinear')
        
        # 4. Compute validation loss using Flow's method
        noise = torch.randn_like(latent)
        _, segment_losses = flow.validation_losses(
            model=model,          
            x1=latent,            
            x0=noise,            
            y=labels,             
            num_segments=8        # From SD3 paper
        )
        
        all_losses.append(segment_losses.cpu())
    
    # 5. Process results and save to CSV
    if len(all_losses) > 0:
        all_losses = torch.stack(all_losses)  # [num_batches, num_segments]
        mean_loss = all_losses.mean().item()
        
        print(f"\nValidation Loss: {mean_loss:.6f}")
        print("Per-segment losses:")
        for i, seg_loss in enumerate(all_losses.mean(dim=0)):
            print(f"Segment {i}: {seg_loss.item():.6f}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['batch_idx'] + [f'segment_{i}' for i in range(8)])
            
            # Write data
            for batch_idx, batch_losses in enumerate(all_losses):
                writer.writerow([batch_idx] + batch_losses.tolist())
            
            # Write mean losses
            writer.writerow(['mean'] + all_losses.mean(dim=0).tolist())
            writer.writerow(['overall_mean', mean_loss])
        
        print(f"Saved losses to {os.path.abspath(save_path)}")
    else:
        print("No validation losses recorded!")


   


if __name__ == "__main__":

    # Example usage:
    # python3 validation_loss/compute_loss.py --output_path validation_loss/output/val_loss_10000.csv --num_samples 10000

    parser = argparse.ArgumentParser(description="Compute validation loss for SiT model.")
    parser.add_argument("--output_path", type=str, default="validation_loss/output/val_loss.csv", help="CSV file to save validation losses.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of validation images to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for validation.")

    args = parser.parse_args()

    val_loader = prepare_dataset(batch_size=args.batch_size, num_samples=args.num_samples)
    module = prepare_model()
    compute_loss(module, val_loader, save_path=args.output_path)
    