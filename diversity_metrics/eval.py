import os
import argparse
import csv
from metrics import DreamSimMetric, LPIPSMetric, DINODiversityMetric, CLIPDiversityMetric,VendiDiversityMetric

def compute_diversity_metrics(npz_path, metric, batch_size):
    if metric == "lpips":
        metric_instance = LPIPSMetric()
    elif metric == "dreamsim":
        metric_instance = DreamSimMetric()
    elif metric == "dinov2":
        metric_instance = DINODiversityMetric()
    elif metric == "clip":
        metric_instance = CLIPDiversityMetric()
    elif metric == "vendi":
        metric_instance = VendiDiversityMetric()
    else:
        raise ValueError("Unsupported metric type.")
    
    return metric_instance.compute_from_npz(npz_path, max_samples=batch_size)

def save_to_csv(csv_path, row_data):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diversity metrics")
    #SiT-XL-2-pretrained-cfg-1.0-4-SDE-250-Euler-sigma-Mean-0.04.npz
    parser.add_argument("--sample_batch", help="path to sample batch npz file",
                        default="/home/stud/xhan/projects/cvprac/SiT/samples/inter_class/num_samples_5000/sde_cfg_1.0/SiT-XL-2-pretrained-cfg-1.0-4-SDE-250-Euler-sigma-Mean-0.04.npz")#SiT-XL-2-pretrained-cfg-1.0-4-ODE-
    parser.add_argument("--metric", type=str, choices=["lpips", "dreamsim", "dinov2", "clip","vendi"], required=True)
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for metric computation")
    parser.add_argument("--csv_path", type=str, default="/home/stud/xhan/projects/cvprac/SiT/diversity_results.csv")
 
    args = parser.parse_args()
    npz_path = args.sample_batch
    mean, std = compute_diversity_metrics(npz_path, args.metric, args.batch_size)

    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

    save_to_csv(
    args.csv_path,
    row_data={
        "metric": args.metric,
        "npz_path": npz_path,
        "batch_size": args.batch_size,
        "mean": f"{mean:.4f}",
        "std": f"{std:.4f}"
    }
)