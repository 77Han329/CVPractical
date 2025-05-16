import lpips
import torch
from diversity_metrics.base_metric import Metric

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LPIPSMetric(Metric):
    """
        Computes the LPIPS perceptual similarity score between two folders of images.
        
        Args:
            net (str): 'alex', 'vgg', or 'squeeze'. Use 'alex' for best forward scores.
    """
    def __init__(self, net='alex'):
        self.loss_fn = lpips.LPIPS(net=net).to(device)

    def name(self):
        return "LPIPS"

    def compute_distance(self, img1, img2):
        img1 = lpips.im2tensor(img1).to(device)
        img2 = lpips.im2tensor(img2).to(device)
        return self.loss_fn(img1, img2).item()
