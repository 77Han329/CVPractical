import torch
import torch.nn as nn

from torchmetrics import SumMetric          # sum over devices
from torchmetrics import MeanMetric         # mean over devices
from torchmetrics.image.fid import FrechetInceptionDistance


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


class ImageMetricTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_samples = SumMetric()

        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        )

    def __call__(self, target, pred):
        """ Assumes target and pred in [-1, 1] range """
        bs = target.shape[0]
        real_ims = un_normalize_ims(target)
        fake_ims = un_normalize_ims(pred)
        
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        self.total_samples.update(bs)

    def reset(self):
        self.fid.reset()
        self.total_samples.reset()

    def aggregate(self):
        """ Compute the final metrics (automatically synced across devices) """
        n_total_samples = int(self.total_samples.compute())
        return {
            f"fid-{n_total_samples}": self.fid.compute(),
            "n_metric_samples": n_total_samples,
        }
