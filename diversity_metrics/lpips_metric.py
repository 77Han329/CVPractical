import os
import itertools
import torch
import lpips
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LPIPSMetric:
    """
    Computes LPIPS perceptual similarity scores between images in a folder or in an .npz file.
    
    Args:
        net (str): 'alex', 'vgg', or 'squeeze'. Use 'alex' for best forward scores.
        use_gpu (bool): Whether to use GPU if available.


    Example usage:
    lpips = LPIPSMetric()
    mean_dist, std_dist = lpips.compute_from_folder("path/to/images")
    print("Mean LPIPS distance (folder):", mean_dist)

    mean_dist, std_dist = lpips.compute_from_npz("path/to/images.npz")
    print("Mean LPIPS distance (npz):", mean_dist)
    """
    def __init__(self, use_gpu=True, net='alex'):
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu and torch.cuda.is_available():
            self.loss_fn = self.loss_fn.to(device)
        self.use_gpu = use_gpu

    def _load_tensor_image(self, image_path):
        # Load and convert image to LPIPS tensor format [1,3,H,W] in [-1,1]
        img = lpips.im2tensor(lpips.load_image(image_path))
        if self.use_gpu and torch.cuda.is_available():
            img = img.to(device)
        return img

    def _preprocess_np_image(self, np_image):
        # Convert np_image [H,W,C] uint8 array to LPIPS tensor
        # LPIPS expects [1,3,H,W] in [-1,1], so we replicate the same conversion
        # lpips.im2tensor converts from PIL.Image, so we replicate manually here:
        if np_image.dtype != np.uint8:
            np_image = (np.clip(np_image, 0, 1)*255).astype(np.uint8)  # normalize float to uint8 if needed
        # Convert to [H,W,3], then to tensor
        img = lpips.im2tensor(np_image)  # lpips.im2tensor accepts np.ndarray as well
        if self.use_gpu and torch.cuda.is_available():
            img = img.to(device)
        return img

    def compute_from_folder(self, folder_path, return_pairwise_scores=False, save_scores_path=None):
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(image_files) < 2:
            raise ValueError("Need at least two images in the folder to compute diversity.")

        tensors = []
        for f in tqdm(image_files, desc="Loading images"):
            path = os.path.join(folder_path, f)
            tensors.append(self._load_tensor_image(path))

        return self._compute_pairwise(tensors, image_files, return_pairwise_scores, save_scores_path)

    def compute_from_npz(self, npz_path, return_pairwise_scores=False, save_scores_path=None):
        data = np.load(npz_path)['arr_0']  # expects shape [N,H,W,C]

        if data.shape[0] < 2:
            raise ValueError("Need at least two images in the npz file to compute diversity.")

        tensors = []
        for i in tqdm(range(data.shape[0]), desc="Loading npz images"):
            tensors.append(self._preprocess_np_image(data[i]))

        image_names = [f"img_{i}" for i in range(data.shape[0])]
        return self._compute_pairwise(tensors, image_names, return_pairwise_scores, save_scores_path)

    def _compute_pairwise(self, tensors, image_names, return_pairwise_scores, save_scores_path):
        pair_scores = {}
        distances = []

        for (i, j) in tqdm(itertools.combinations(range(len(tensors)), 2),
                           desc="Computing pairwise LPIPS",
                           total=len(tensors) * (len(tensors) - 1) // 2):
            with torch.no_grad():
                dist = self.loss_fn(tensors[i], tensors[j]).item()
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
