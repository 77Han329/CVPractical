# pip install lpips
# pip install opencv-python

import lpips
import torch
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class LPIPS:
    """
    Computes the LPIPS perceptual similarity score between two folders of images.
    
    Args:
        net (str): 'alex', 'vgg', or 'squeeze'. Use 'alex' for best forward scores.

    Example call:
   
        lpips = LPIPS()
        avg_score = lpips.compute(
            '/path/to/folderA',
            '/path/to/folderB',
            # save_scores='results/lpips_scores.txt'
        )
        print(f"Average LPIPS: {avg_score}")

    """
    def __init__(self, net='alex', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)

    def _load_image(self, path):
        image = lpips.im2tensor(lpips.load_image(path))  # Converts to tensor in [-1, 1]
        return image.to(self.device)

    def compute(self, folder1, folder2, save_scores=None):
        """
        Computes LPIPS scores for corresponding images in two folders.

        Args:
            folder1 (str): Path to the first folder.
            folder2 (str): Path to the second folder.
            save_scores (str or None): If provided, path to save the per-image scores.
        
        Returns:
            dict: Dictionary mapping filenames to LPIPS scores.
        """
        files1 = sorted(os.listdir(folder1))
        files2 = sorted(os.listdir(folder2))

        assert len(files1) == len(files2), "Folders must contain the same number of images."

        scores = {}
        if save_scores:
            os.makedirs(os.path.dirname(save_scores), exist_ok=True)
            f = open(save_scores, 'w')

        for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Computing LPIPS"):
            path1 = os.path.join(folder1, file1)
            path2 = os.path.join(folder2, file2)

            if not os.path.exists(path1) or not os.path.exists(path2):
                print(f"For {file1} or {file2}, image missing.")
                break

            try:
                img0 = self._load_image(path1)
                img1 = self._load_image(path2)
                dist = self.loss_fn(img0, img1).item()
                scores[file1] = dist
                if save_scores:
                    f.write(f"{file1}: {dist:.6f}\n")
            except Exception as e:
                print(f"Error processing {file1}: {e}")

        if save_scores:
            f.close()

        return sum(scores.values()) / (len(scores) + 1e-8)
    


