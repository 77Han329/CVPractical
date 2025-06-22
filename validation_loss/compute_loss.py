#%pip install torchvision -q
#%pip install pytorch-lightning -q
#%pip install torchmetrics[image] -q

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(batch_size=64):
    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    class ImageNetTorchDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset = hf_dataset
            self.transform = transform

        def __getitem__(self, idx):
            image = self.dataset[idx]['image']
            image = self.transform(image)
            return image

        def __len__(self):
            return len(self.dataset)

    dataset = load_dataset("mlx-vision/imagenet-1k", split="validation")
    val_dataset = ImageNetTorchDataset(dataset, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
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

    cfg = OmegaConf.load("dit-xl-2.yaml")
    module = TrainerModuleLatentFlow(
        model=model,
        flow_cfg=cfg
    )   
    module.eval()
    module.to(device)
    print("Model prepared and loaded with pre-trained weights.")
    return module


def compute_loss(module, val_loader, device="cuda", save_path="validation_loss/output/val_loss.npy"):
    print("Running validation...")
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
        batch = batch.to(device)
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

if __name__ == "__main__":
    batch_size = 64
    val_loader = prepare_dataset(batch_size)
    module = preprare_model()
    
    compute_loss(module, val_loader)