from diversity_metrics.base_metric import Metric
from dreamsim import dreamsim
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DreamSimMetric(Metric):
    """
    Perceptual distance between two images. A higher score means more different, lower means more similar.

    Args: 
        pretrained(str): "ensemble", "dino_vitb16", "open_clip_vitb32", "clip_vitb32", "dinov2_vitb14", "synclr_vitb16", "dino_vitb16", "dinov2_vitb14"

    """
    def __init__(self, pretrained=True):
        self.model, self.preprocess = dreamsim(pretrained=pretrained, device=device)

    def name(self):
        return "DreamSim"

    def compute_distance(self, img1, img2):
        img1 = self.preprocess(img1).unsqueeze(0).to(device)
        img2 = self.preprocess(img2).unsqueeze(0).to(device)
        return self.model(img1, img2).item()
