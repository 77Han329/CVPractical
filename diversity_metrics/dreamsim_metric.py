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
    Computes DreamSim perceptual similarity scores between images
    from either a folder or a .npz file containing images.

    Usage:
        metric = DreamSimMetric()
        mean_dist, std_dist = metric.compute("path/to/folder_or_npz")
        print("Mean DreamSim Distance:", mean_dist)
    """
    def __init__(self, pretrained=True):
        self.model, self.preprocess = dreamsim(pretrained=pretrained, device=device)
        self.model.eval()

    def _preprocess_np_image(self, np_image):
        img = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")
        return self.preprocess(img).to(device)

    def _load_images_from_folder(self, folder_path):
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if len(image_files) < 2:
            raise ValueError("Need at least two images in the folder to compute diversity.")

        tensors = []
        for f in tqdm(image_files, desc="Loading images from folder"):
            img = Image.open(os.path.join(folder_path, f)).convert("RGB")
            tensors.append(self.preprocess(img).to(device))
        return tensors, image_files

    def _load_images_from_npz(self, npz_path):
        data = np.load(npz_path)['arr_0']  # expects shape [N,H,W,C]
        if data.shape[0] < 2:
            raise ValueError("Need at least two images in npz to compute diversity.")

        tensors = []
        for i in tqdm(range(data.shape[0]), desc="Loading images from npz"):
            tensors.append(self._preprocess_np_image(data[i]))
        image_names = [f"img_{i}" for i in range(data.shape[0])]
        return tensors, image_names

    def compute(self, input_path, return_pairwise_scores=False, save_scores_path=None):
        """
        Automatically detects if input_path is a folder or npz file
        and computes DreamSim diversity scores.
        """
        if os.path.isdir(input_path):
            tensors, image_names = self._load_images_from_folder(input_path)
        elif input_path.endswith('.npz'):
            tensors, image_names = self._load_images_from_npz(input_path)
        else:
            raise ValueError("Input path must be a folder or a .npz file.")

        pair_scores = {}
        distances = []

        for (i, j) in tqdm(itertools.combinations(range(len(tensors)), 2),
                           desc="Computing pairwise DreamSim",
                           total=len(tensors) * (len(tensors) - 1) // 2):
            with torch.no_grad():
                dist = self.model(tensors[i], tensors[j]).item()
            pair_name = f"{image_names[i]}___{image_names[j]}"
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
