import os
import argparse
import csv
from metrics import DreamSimMetric, LPIPSMetric, DINODiversityMetric, CLIPDiversityMetric,VendiDiversityMetric

def compute_diversity_metrics(npz_path, metric):
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
    
    return metric_instance.compute_from_npz(npz_path)

def preprocess_path(number_of_samples, label, sample_method, cfg, sample_steps, diffusion_form):
    base_dir = "/home/stud/xhan/projects/cvprac/SiT/samples"
    setting_path = os.path.join(base_dir, 
                                f"num_samples_{number_of_samples}", 
                                f"label_{label}",
                                f"{sample_method}_cfg_{cfg}")
    
    if sample_method == "ode":
        npz_path = os.path.join(setting_path, 
                                f"SiT-XL-2-pretrained-cfg-{cfg}-4-ODE-{sample_steps}-dopri5.npz")
    elif sample_method == "sde":
        npz_path = os.path.join(setting_path,
                                f"{diffusion_form}",
                                f"SiT-XL-2-pretrained-cfg-{cfg}-4-SDE-{sample_steps}-Euler-{diffusion_form}-Mean-0.04.npz")
    else:
        raise ValueError("Unknown sample method.")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f" File not found: {npz_path}")
    
    return npz_path

def save_to_csv(csv_path, row_data, header=None):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists and header:
            writer.writeheader()
        writer.writerow(row_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diversity metrics")
    parser.add_argument("--metric", type=str, choices=["lpips", "dreamsim", "dinov2", "clip","vendi"], required=True)
    parser.add_argument("--number-of-samples", type=int, default=100)
    parser.add_argument("--label", type=int, choices=[0, 97, 300, 389, 409, 555, 569, 571, 574, 701], default=0)
    parser.add_argument("--sample-method", type=str, choices=["ode", "sde"], default="ode")
    parser.add_argument("--cfg", type=float, choices=[1.0, 3.0, 5.0], default=1.0)
    parser.add_argument("--sample-steps", type=int, choices=[50, 100, 150, 200, 250], default=250)
    parser.add_argument("--diffusion-form", type=str, choices=["sigma", "increase"], default="sigma")
    parser.add_argument("--csv-path", type=str, default="diversity_results.csv")

    args = parser.parse_args()
    npz_path = preprocess_path(args.number_of_samples, args.label, args.sample_method, args.cfg, args.sample_steps, args.diffusion_form)
    mean, std = compute_diversity_metrics(npz_path, args.metric)

    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

    save_to_csv(
        args.csv_path,
        row_data={
            "metric": args.metric,
            "label": args.label,
            "cfg": args.cfg,
            "sample_steps": args.sample_steps,
            "sample_method": args.sample_method,
            "diffusion_form": args.diffusion_form,
            "num_samples": args.number_of_samples,
            "mean": f"{mean:.4f}",
            "std": f"{std:.4f}"
        },
        header=["metric", "label", "cfg", "sample_steps", "sample_method", "diffusion_form", "num_samples", "mean", "std"]
    )