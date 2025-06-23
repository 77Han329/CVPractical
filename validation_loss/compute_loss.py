#%pip install torchvision -q
#%pip install pytorch-lightning -q
#%pip install torchmetrics[image] -q

# run with `python -m validation_loss.image_ldm_main.ldm.logging`

import torch
import torchvision
import os
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from train import center_crop_arr
from models import SiT_models
from download import find_model
from validation_loss.image_ldm_main.ldm.trainer import TrainerModuleLatentFlow
import torch.nn.functional as F
from validation_loss.image_ldm_main.ldm.flow import Flow
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io

class ImageNetValDatasetWithLabels(Dataset):
    def __init__(self, image_dir, label_txt_path, meta_mat_path, synset_words_path, transform=None, max_images=1000):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*")))[:max_images]
        print(len(self.image_paths), "images found in", image_dir)
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


def prepare_dataset(batch_size=64):
    #dataset = load_dataset("mlx-vision/imagenet-1k", split="validation")

    val_dataset = ImageNetValDatasetWithLabels(
        image_dir="/home/coder/ILSVRC2012_img_val",
        label_txt_path="validation_loss/preprocessing/data/ILSVRC2012_validation_ground_truth.txt",
        meta_mat_path="validation_loss/preprocessing/data/meta.mat",
        synset_words_path="validation_loss/preprocessing/data/synset_words.txt"
    )

    #val_subset = torch.utils.data.Subset(val_dataset, indices=range(1_000)) 
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False,
    )
    print(f"Validation set prepared with {len(val_loader.dataset)} images.")
    return val_loader


def preprare_model():
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

"""
def compute_loss(module, val_loader, save_path="validation_loss/output/val_loss.npy"):
    print("Running validation...")
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
        #batch = batch.to(device)
        module.validation_step(batch, batch_idx=batch_idx)

    # --- Compute validation loss (aggregate)
    module.on_validation_epoch_end()

    if hasattr(module, "val_losses") and hasattr(module.val_losses, "compute"):
        if len(module.val_losses.value) > 0:
            val_loss = module.val_losses.compute().cpu()
            val_loss_np = val_loss.numpy()

            # Print mean loss
            mean_loss = val_loss_np.mean()
            print(f"\n Validation Loss (mean): {mean_loss:.6f}")

            # Save full loss array
            np.save(save_path, val_loss_np)
            print(f"Saved full loss to {os.path.abspath(save_path)}")
        else:
            print("No validation losses recorded — check if validation_step populates val_losses.")
    else:
        print("module.val_losses is not properly initialized.")
"""

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
            if latent.shape[-2:] != (32, 32):
                latent = F.interpolate(latent, size=(32, 32), mode='bilinear')
        
        # 4. Compute validation loss using Flow's method
        noise = torch.randn_like(latent)
        _, segment_losses = flow.validation_losses(
            model=model,          # Your SiT model
            x1=latent,            # Real data
            x0=noise,             # Noise
            y=labels,             # Optional labels
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
    batch_size = 16
    val_loader = prepare_dataset(batch_size)
    module = preprare_model()
    
    compute_loss(module, val_loader)

