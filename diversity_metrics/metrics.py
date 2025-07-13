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

from vendi_score import vendi, image_utils
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms


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
        mean_dist, std_dist = lpips.compute_from_npz("path/to/images.npz", max_samples=1000)
    """
    def __init__(self, use_gpu=True, net='alex'):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.use_gpu = use_gpu

    # def _load_tensor_image(self, image_path):
    #     img = lpips.im2tensor(lpips.load_image(image_path))
    #     return img.to(self.device) if self.use_gpu and torch.cuda.is_available() else img

    def _preprocess_np_image(self, np_image):
        if np_image.dtype != np.uint8:
            np_image = (np.clip(np_image, 0, 1) * 255).astype(np.uint8)
        img = lpips.im2tensor(np_image)
        return img.to(self.device) if self.use_gpu and torch.cuda.is_available() else img
    
    def compute_from_npz(self, npz_path, return_pairwise_scores=False, save_scores_path=None,
                     max_samples=10, random_sample=True, seed=10):
        data = np.load(npz_path)['arr_0']

        if data.shape[0] < 2:
            raise ValueError("Need at least two images in the npz file to compute diversity.")

        if max_samples is not None and max_samples < data.shape[0]:
            if random_sample:
                rng = np.random.default_rng(seed)
                indices = rng.choice(data.shape[0], size=max_samples, replace=False)
                data = data[indices]
            else:
                data = data[:max_samples]

        tensors = []
        for i in tqdm(range(data.shape[0]), desc="Loading npz images"):
            tensors.append(self._preprocess_np_image(data[i]))

        image_names = [f"img_{i}" for i in range(len(tensors))]
        return self._compute_pairwise(tensors, image_names, return_pairwise_scores, save_scores_path)


    # def compute_from_npz(self, npz_path, return_pairwise_scores=False, save_scores_path=None, max_samples=10):
    #     data = np.load(npz_path)['arr_0']

    #     if data.shape[0] < 2:
    #         raise ValueError("Need at least two images in the npz file to compute diversity.")

    #     if max_samples is not None:
    #         data = data[:max_samples]

    #     tensors = []
    #     for i in tqdm(range(data.shape[0]), desc="Loading npz images"):
    #         tensors.append(self._preprocess_np_image(data[i]))

    #     image_names = [f"img_{i}" for i in range(len(tensors))]
    #     return self._compute_pairwise(tensors, image_names, return_pairwise_scores, save_scores_path)

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

    def compute_from_npz(self, input_path, return_pairwise_scores=False, save_scores_path=None):
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


# ==================== DINOv2 ====================
class DINODiversityMetric:
    def __init__(self, use_gpu=True):
        from PIL import Image
        import torchvision.transforms as T

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

    def _preprocess_np_image(self, np_img):
        from PIL import Image
        if np_img.dtype != np.uint8:
            np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def compute_from_npz(self, npz_path):
        import numpy as np
        from tqdm import tqdm
        data = np.load(npz_path)["arr_0"]
        feats = []

        print("Extracting DINOv2 features (via HuggingFace)...")
        for img in tqdm(data):
            pil_img = self._preprocess_np_image(img)
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                feat = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            feats.append(feat.squeeze(0))

        feats = torch.stack(feats)
        feats = F.normalize(feats, dim=1)

        dists = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                dists.append(1.0 - torch.dot(feats[i], feats[j]).item())  # cosine

        return float(np.mean(dists)), float(np.std(dists))


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
            mean, std = metric.compute_from_npz("test.npz")

            # Average pooled features
            metric = CLIPDiversityMetric(feature_type='avg_pool')
            mean, std = metric.compute_from_npz("test.npz")
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if feature_type not in ['cls_token', 'avg_pool']:
            raise ValueError("feature_type must be either 'cls_token' or 'avg_pool'")
        self.feature_type = feature_type
    
    def _preprocess_np_image(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(np_img)
    
    def _get_features(self, img):
        """Get features based on selected feature_type"""
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.feature_type == 'cls_token':
                # Standard CLIP features
                feat = self.model.get_image_features(**inputs)
            else:
                # Average pooled features
                outputs = self.model.vision_model(**inputs)
                patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Exclude CLS token
                avg_pooled = patch_embeddings.mean(dim=1)
                feat = self.model.visual_projection(avg_pooled)
            
            return F.normalize(feat, dim=1).squeeze(0)
    
    def compute_from_npz(self, npz_path):
        data = np.load(npz_path)["arr_0"]
        feats = []

        print(f"Extracting {self.feature_type} features...")
        for img in tqdm(data):
            pil_img = self._preprocess_np_image(img)
            feats.append(self._get_features(pil_img))

        feats = torch.stack(feats)
        dists = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                dists.append(1.0 - torch.dot(feats[i], feats[j]).item())  # cosine distance

        return float(np.mean(dists)), float(np.std(dists))


# ==================== Vendi Diversity ====================

class VendiDiversityMetric:
    """
    Computes Vendi diversity scores (pixel-level or feature-level).

    Requirements:
        If using 'inception' type, make sure ~/.env/lib/python3.11/site-packages/vendi_score/image_utils.py includes:
            from torchvision.models.inception import Inception_V3_Weights
    """
    def __init__(self, type="inception", use_gpu=True):
        """
        Args:
            type (str): 'inception' or 'pixel' (default: 'inception')
            use_gpu (bool): Whether to use GPU if available
        """
        if type not in ("inception", "pixel"):
            raise ValueError("type must be either 'inception' or 'pixel'")
        self.type = type
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    def _to_pil(self, img):
        if img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8), mode="L")
        elif img.ndim == 3:
            return Image.fromarray(img.astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

    def _load_images_from_npz(self, npz_path):
        """Load images from npz file and convert to PIL."""
        data = np.load(npz_path)["arr_0"]
        if data.shape[0] < 2:
            raise ValueError("Need at least two images to compute Vendi diversity.")
        return [self._to_pil(img) for img in data]

    def compute_from_npz(self, npz_path):
        """Compute Vendi diversity score from a .npz file."""
        images = self._load_images_from_npz(npz_path)

        if self.type == "pixel":
            score = image_utils.pixel_vendi_score(images)
        elif self.type == "inception":
            score = image_utils.embedding_vendi_score(images, device=self.device)
        else:
            raise RuntimeError("Unsupported feature type")

        return float(score), None
    

# ==================== MS-SSIM Diversity ====================

# pip install monai
class SSIMDiversityMetric:
    """ Computes pairwise SSIM diversity scores from a .npz file."""
    def __init__(self, use_gpu=True, data_range=1.0):
        """
        Args:
            use_gpu: Use GPU if available
            data_range: 1.0 for [0,1], 255.0 for [0,255]
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=11  # default size for SSIM
        ).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def compute_from_npz(self, npz_path):
        """Compute pairwise SSIM diversity"""
        data = np.load(npz_path)["arr_0"]
        tensors = [self.transform(Image.fromarray((x*255).astype(np.uint8)))
                  .unsqueeze(0).to(self.device) for x in data]
        
        scores = []
        for i in tqdm(range(len(tensors))):
            for j in range(i+1, len(tensors)):
                with torch.no_grad():
                    score = self.ssim(tensors[i], tensors[j]).item()
                scores.append(score)
        
        mean_div = float(np.mean(scores))
        std_div = float(np.std(scores))
        #print(f"Mean: {mean_div:.4f}, Std: {std_div:.4f}")
        return mean_div, std_div