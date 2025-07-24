import os
import argparse
import csv
import re
from metrics import DreamSimMetric, LPIPSMetric, DINODiversityMetric, CLIPDiversityMetric, VendiDiversityMetric, SSIMDiversityMetric

def compute_diversity_metrics(npz_path, metric, batch_size, seed, feature_type):
    """
    Compute diversity metrics from a given npz file.
    
    Args:
        npz_path (str): Path to the npz file containing samples.
        metric (str): Type of metric to compute ('lpips', 'dreamsim', 'dinov2', 'clip').
        batch_size (int): Number of samples to process in each batch.
        seed (int): Random seed for reproducibility.
        feature_type (str): Feature type for clip/dino metrics ('cls_token' or 'avg_pool').
    
    Returns:
        tuple: Mean and standard deviation of the computed metric.
    """
    if metric == "lpips":
        metric_instance = LPIPSMetric()
    elif metric == "dreamsim":
        metric_instance = DreamSimMetric()
    elif metric == "dinov2":
        metric_instance = DINODiversityMetric(feature_type=feature_type)
    elif metric == "clip":
        metric_instance = CLIPDiversityMetric(feature_type=feature_type)
    elif metric == "vendi":
        metric_instance = VendiDiversityMetric()
    elif metric == "ssim":
        metric_instance = SSIMDiversityMetric()
    else:
        raise ValueError("Unsupported metric type.")
    
    return metric_instance.compute_from_npz(npz_path, max_samples=batch_size,seed=seed)

def save_to_csv(csv_path, row_data):
    """
    Save metric results to a CSV file.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def extract_setting_from_path(npz_path):
    """
    Extracts a simplified setting name from the npz filename.
    e.g. 'SiT-XL-2-pretrained-cfg-1.5-4-ODE-250-euler.npz' => 'cfg-1.5-ODE'
    """
    filename = os.path.basename(npz_path)
    match = re.search(r'cfg-(\d\.\d)-\d+-(ODE|SDE)', filename)
    if match:
        return f"cfg-{match.group(1)}-{match.group(2)}"
    else:
        return "unknown_setting"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diversity metrics")
    parser.add_argument("--sample_batch", help="path to sample batch npz file", default="SiT-XL-2-pretrained-cfg-1.0-4-ODE-250-euler.npz")
    parser.add_argument("--metric", type=str, choices=["lpips", "dreamsim", "dinov2", "clip"], required=True)
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for metric computation",required=True)
    parser.add_argument("--feature_type", type=str, choices=["cls_token", "avg_pool"],
                        default="cls_token", help="Feature type for clip/dino metrics")
    parser.add_argument("--csv_path", type=str,help="Path to save the CSV results", default="diversity_metrics_results.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    assert args.batch_size < 1000, "Batch size should be less than 1000 to avoid memory issues."
    npz_path = args.sample_batch
    seed = args.seed
    mean, std = compute_diversity_metrics(npz_path, args.metric, args.batch_size,seed,args.feature_type)


    print(f"Metric: {args.metric} | Setting: {extract_setting_from_path(npz_path)}")
    print(f"Mean: {mean:.4f} | Std: {std:.4f}")

    # Save result
    save_to_csv(
        args.csv_path,
        row_data={
            "setting": extract_setting_from_path(npz_path),
            "path": npz_path,
            "metric": args.metric,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "feature_type": args.feature_type,
            "mean": f"{mean:.4f}",
            "std": f"{std:.4f}"
        }
    )
