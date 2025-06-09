from diversity_metrics.dreamsim_metric import DreamSimMetric
from diversity_metrics.lpips_metric import LPIPSMetric

metric = DreamSimMetric()
mean_dist, std_dist = metric.compute("test/img_npz.npz")
print("Mean DreamSim Distance:", mean_dist)

lpips = LPIPSMetric()
mean_dist, std_dist = lpips.compute_from_npz("test/img_npz.npz")
print("Mean LPIPS distance (npz):", mean_dist)