# diversity_metrics.py

import os
import itertools
import torch
import lpips
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from dreamsim import dreamsim
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import torch.nn.functional as F
import random

# ==================== Utility Functions ====================
def sample_npz_images(npz_array, max_samples=None, random_sample=True,seed=42):
    """
    Subsamples an array of images from .npz.

    Args:
        npz_array (np.ndarray): shape (N, H, W, C)
        max_samples (int): max number of samples to use
        random_sample (bool): sample randomly or take first N
        seed (int): random seed

    Returns:
        np.ndarray: subsampled array
    """
    if max_samples is None or max_samples >= len(npz_array):
        return npz_array
    if random_sample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(npz_array), size=max_samples, replace=False)
        return npz_array[indices]
    else:
        return npz_array[:max_samples]
    
def cosine_similarity(feats):
    """
    Computes pairwise cosine similarity between features.

    Args:
        feats (torch.Tensor): shape (N, D)
    Returns:
        tuple: (mean_distance, std_distance)
        
    """
    dists = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            dists.append(1.0 - torch.dot(feats[i], feats[j]).item())  # cosine distance

    return float(np.mean(dists)), float(np.std(dists))

def get_features(model, processor, data, device, feature_type='cls_token'):
    feats = []
    is_clip = hasattr(model, "vision_model") and hasattr(model, "get_image_features")

    for img in tqdm(data):
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if is_clip:
                if feature_type == 'cls_token':
                    feat = model.get_image_features(**inputs)
                elif feature_type == 'avg_pool':
                    outputs = model.vision_model(**inputs)
                    patch_embeddings = outputs.last_hidden_state[:, 1:, :]
                    avg_pooled = patch_embeddings.mean(dim=1)
                    feat = model.visual_projection(avg_pooled)
                else:
                    raise ValueError("Invalid feature_type for CLIP.")
            else:  # assume DINO
                outputs = model(**inputs)
                if feature_type == 'cls_token':
                    feat = outputs.last_hidden_state[:, 0, :]
                elif feature_type == 'avg_pool':
                    patch_embeddings = outputs.last_hidden_state[:, 1:, :]
                    feat = patch_embeddings.mean(dim=1)
                else:
                    raise ValueError("Invalid feature_type for DINO.")

        feats.append(feat.squeeze(0))

    feats = torch.stack(feats)
    feats = F.normalize(feats, dim=1)
    return feats
# ==================== LPIPS ====================
class LPIPSMetric:
    """
    Computes LPIPS perceptual similarity scores between images in a folder or in an .npz file.

    Args:
        net (str): 'alex', 'vgg', or 'squeeze'. Use 'alex' for best forward scores.
        use_gpu (bool): Whether to use GPU if available.

    Example usage:
        lpips = LPIPSMetric()
        mean_dist, std_dist = lpips.compute_from_folder("path/to/images", max_samples=1000)
        mean_dist, std_dist = lpips.compute_div("path/to/images.npz", max_samples=1000)
    """
    def __init__(self, use_gpu=True, net='alex'):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.use_gpu = use_gpu

    def _preprocess_np_image(self, np_image):
        if np_image.dtype != np.uint8:
            np_image = (np.clip(np_image, 0, 1) * 255).astype(np.uint8)
        img = lpips.im2tensor(np_image)
        return img.to(self.device) if self.use_gpu and torch.cuda.is_available() else img
    
    def compute_div(self, npz_path, return_pairwise_scores=False, save_scores_path=None,
                     max_samples=10, random_sample=True, seed=10):
        
        data = np.load(npz_path)["arr_0"]
        data = sample_npz_images(data, max_samples, random_sample, seed)
        

        if data.shape[0] < 2:
            raise ValueError("Need at least two images in the npz file to compute diversity.")


        tensors = []
        for i in tqdm(range(data.shape[0]), desc="Loading npz images"):
            tensors.append(self._preprocess_np_image(data[i]))

        image_names = [f"img_{i}" for i in range(len(tensors))]
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

        return (avg_dist, std_dist, pair_scores) if return_pairwise_scores else (avg_dist, std_dist)

# ==================== DreamSim ====================
class DreamSimMetric:
    """
    Computes DreamSim perceptual similarity scores between images in a .npz file.

    Args:
        pretrained (bool): Whether to load pretrained DreamSim weights

    Usage:
        metric = DreamSimMetric()
        mean_dist, std_dist = metric.compute_div("images.npz", max_samples=100)
    """
    def __init__(self, pretrained=True, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = dreamsim(pretrained=pretrained, device=self.device)
        self.model.eval()

    def _preprocess_np_image(self, np_image):
        img = Image.fromarray(np_image.astype(np.uint8)).convert("RGB")
        return self.preprocess(img).to(self.device)

    def compute_div(self, npz_path, return_pairwise_scores=False, save_scores_path=None,
                         max_samples=10, random_sample=True, seed=42):
        """
        Computes pairwise DreamSim distances from .npz file

        Args:
            npz_path (str): Path to .npz file
            return_pairwise_scores (bool): Whether to return individual pair scores
            save_scores_path (str): Optional file path to save pair scores
            max_samples (int): Number of samples to use (default 10)
            random_sample (bool): Whether to randomly sample
            seed (int): Random seed

        Returns:
            tuple: (mean_distance, std_distance) or (mean, std, pair_scores)
        """
        data = np.load(npz_path)['arr_0']
        data = sample_npz_images(data, max_samples, random_sample, seed)

        if data.shape[0] < 2:
            raise ValueError("Need at least two images in the npz file to compute diversity.")

        tensors = [self._preprocess_np_image(img) for img in tqdm(data, desc="Preprocessing DreamSim")]
        image_names = [f"img_{i}" for i in range(len(tensors))]

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

        return (avg_dist, std_dist, pair_scores) if return_pairwise_scores else (avg_dist, std_dist)

# ==================== DINOv2 ====================
class DINODiversityMetric:
    def __init__(self, use_gpu=True,feature_type='avg_pool'):

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
        if feature_type not in ['cls_token', 'avg_pool']:
            raise ValueError("feature_type must be either 'cls_token' or 'avg_pool'")
        self.feature_type = feature_type


    def compute_div(self, npz_path,max_samples=10, random_sample=True, seed=5):
        
        data = np.load(npz_path)["arr_0"]
        data = sample_npz_images(data, max_samples, random_sample, seed)
        
        if data.shape[0] < 2:
            raise ValueError("Need at least two images in the npz file to compute diversity.")

        print(f"Extracting DINOv2 features from {len(data)} images...")
        feats = get_features(self.model, self.processor, data, self.device, self.feature_type)
        
        mean, std = cosine_similarity(feats)

        return mean, std

# ==================== CLIP Diversity ====================
class CLIPDiversityMetric:
    def __init__(self, use_gpu=True, feature_type='cls_token'):
        """
        Args:
            use_gpu: Whether to use GPU if available
            feature_type: Either 'cls_token' (default) or 'avg_pool'

        Example usage:
            # CLS token features (default)
            metric = CLIPDiversityMetric()
            mean, std = metric.compute_div("test.npz")

            # Average pooled features
            metric = CLIPDiversityMetric(feature_type='avg_pool')
            mean, std = metric.compute_div("test.npz")
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if feature_type not in ['cls_token', 'avg_pool']:
            raise ValueError("feature_type must be either 'cls_token' or 'avg_pool'")
        self.feature_type = feature_type
        
    
    
    def compute_div(self, npz_path,max_samples=10, random_sample=True, seed=4):
        
        data = np.load(npz_path)["arr_0"]
        data = sample_npz_images(data, max_samples, random_sample, seed)
        
        if data.shape[0] < 2:
            raise ValueError("Need at least two images in the npz file to compute diversity.")

        print(f"Extracting Clip features from {len(data)} images...")
        feats = get_features(self.model, self.processor, data, self.device, self.feature_type)
        
        mean, std = cosine_similarity(feats)

        return mean, std



# ==================== Vendi Diversity ====================

# pip install vendi_score
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from sklearn.preprocessing import normalize
# from PIL import Image
# from tqdm import tqdm
# import vendi_score.vendi as vendi
# from torchvision.models import inception_v3, Inception_V3_Weights


# class NumpyImageDataset(Dataset):
#     def __init__(self, array, transform=None):
#         self.array = array
#         self.transform = transform

#     def __len__(self):
#         return len(self.array)

#     def __getitem__(self, idx):
#         img = Image.fromarray(self.array[idx].astype(np.uint8))
#         if self.transform:
#             img = self.transform(img)
#         return img
