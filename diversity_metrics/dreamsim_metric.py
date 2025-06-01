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
    Computes DreamSim perceptual similarity scores between images from a .npz file.
    The .npz file must contain an array under the key 'arr_0' of shape [N, H, W, C].

    Example call from another file in main folder structure:

    from diversity_metrics.dreamsim_metric import DreamSimMetric

    deramsim = DreamSimMetric()
    mean_dist, std_dist = deramsim.compute("path/tp/npz_file.npz")
    print("Mean DreamSim Distance:", mean_dist)

    """
    def __init__(self, pretrained=True):
        self.model, self.preprocess = dreamsim(pretrained=pretrained, device=device)
        self.model.eval()

    def _preprocess_np_image(self, np_image):
        img = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")
        return self.preprocess(img).to(device)

    def compute(self, npz_path, return_pairwise_scores=False, save_scores_path=None):
        data = np.load(npz_path)['arr_0']  # Assumes key is 'arr_0'
        if data.shape[0] < 2:
            raise ValueError("Need at least two images to compute diversity.")
        
        tensors = []
        for i in tqdm(range(data.shape[0]), desc="Loading images"):
            tensors.append(self._preprocess_np_image(data[i]))

        pair_scores = {}
        distances = []

        for (i, j) in tqdm(itertools.combinations(range(len(tensors)), 2),
                           desc="Computing pairwise DreamSim",
                           total=len(tensors) * (len(tensors)-1) // 2):
            with torch.no_grad():
                dist = self.model(tensors[i], tensors[j]).item()
            pair_name = f"img_{i}___img_{j}"
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
