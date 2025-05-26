import os
import itertools
import torch
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
from diversity_metrics.base_metric import Metric

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LPIPSMetric():
    """
        Computes the LPIPS perceptual similarity score between two folders of images.
        
        Args:
            net (str): 'alex', 'vgg', or 'squeeze'. Use 'alex' for best forward scores.
    """
    def __init__(self, use_gpu=True, net='alex'):
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu and torch.cuda.is_available():
            self.loss_fn = self.loss_fn.cuda()
        self.use_gpu = use_gpu

    def _load_tensor_image(self, image_path):
        img = lpips.im2tensor(lpips.load_image(image_path))  # Converts to [-1, 1], shape [1,3,H,W]
        if self.use_gpu:
            img = img.cuda()
        return img

    def compute_diversity(self, folder_path, return_pairwise_scores=False, save_scores_path=None):
        image_files = sorted([
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        tensors = []
        for f in tqdm(image_files, desc="Loading images"):
            path = os.path.join(folder_path, f)
            tensors.append(self._load_tensor_image(path))
        
        pair_scores = {}
        distances = []

        for (i, j) in tqdm(itertools.combinations(range(len(tensors)), 2), 
                           desc="Computing pairwise LPIPS", 
                           total=len(tensors) * (len(tensors)-1) // 2):
            dist = self.loss_fn.forward(tensors[i], tensors[j]).item()
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
