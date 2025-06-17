import requests
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from models import SiT_models

device = "cuda" if torch.cuda.is_available() else "cpu"


# --- get data ---

dataset = load_dataset("mlx-vision/imagenet-1k", split="validation")

# Preprocessing: Resize, CenterCrop, Normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
])

# Wrap it in a PyTorch-compatible Dataset
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

val_dataset = ImageNetTorchDataset(dataset, val_transform)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)


"""
# --- download model ---

url = "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0"
response = requests.get(url)

# Write to a local file
with open("SiT-XL-2-256.pt", "wb") as f:
    f.write(response.content)
"""

# --- load model ---


model = "SiT-XL/2"
num_classes = 1000
image_size = 256
learn_sigma = image_size == 256

latent_size = image_size // 8
model = SiT_models["SiT-XL/2"](
    input_size=latent_size,
    num_classes=1000,
    learn_sigma=learn_sigma,
)

model.eval()
model.to(device)


# --- compute val loss ---

from validation_loss.image_ldm_main.ldm.trainer import on_validation_epoch_end, TrainerModuleLatentFlow

module = TrainerModuleLatentFlow(model=model)
module.eval()
module.to(device)

# --- Run validation on batches
for batch_idx, batch in enumerate(val_loader):
    batch = batch.to(device)
    module.validation_step(batch, batch_idx=batch_idx)

# --- Compute validation loss (aggregate)
module.on_validation_epoch_end()

val_loss = module.val_losses.compute()
val_loss_list = val_loss.cpu().tolist()

import json
with open("val_loss.json", "w") as f:
    json.dump({"val_loss": val_loss_list}, f, indent=2)
