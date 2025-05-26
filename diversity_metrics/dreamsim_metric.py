import os
import itertools
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from dreamsim import dreamsim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DreamSimMetric:
    """
    Computes DreamSim perceptual similarity scores between images in a folder.

    Args:
        pretrained (str or bool): DreamSim model name or True for ensemble.
    """
    def __init__(self, pretrained=True):
        self.model, self.preprocess = dreamsim(pretrained=pretrained, device=device)
        self.model.eval()

    def _load_tensor_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return self.preprocess(img).to(device)

    def compute_diversity(self, folder_path, return_pairwise_scores=False, save_scores_path=None):
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(image_files) < 2:
            raise ValueError("Need at least two images to compute diversity.")

        tensors = []
        for f in tqdm(image_files, desc="Loading images"):
            path = os.path.join(folder_path, f)
            tensors.append(self._load_tensor_image(path))

        pair_scores = {}
        distances = []

        for (i, j) in tqdm(itertools.combinations(range(len(tensors)), 2),
                           desc="Computing pairwise DreamSim",
                           total=len(tensors) * (len(tensors)-1) // 2):
            with torch.no_grad():
                dist = self.model(tensors[i], tensors[j]).item()
            pair_name = f"{image_files[i]}___{image_files[j]}"
            pair_scores[pair_name] = dist
            distances.append(dist)

        avg_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))

        if save_scores_path:
            os.makedirs(os.path.dirname(save_scores_path), exist_ok=True)
            with open(save_scores_path, "w") as f:
                for name, score in pair_scores.items():
                    f.write(f"{name}: {score:.6f}\n")

        if return_pairwise_scores:
            return avg_dist, std_dist, pair_scores
        else:
            return avg_dist, std_dist
