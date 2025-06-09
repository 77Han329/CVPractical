import os
import argparse
from metrics import DreamSimMetric, LPIPSMetric,DINODiversityMetric,DINOKDDMetric

def compute_diversity_metrics(npz_path, metric):
    if metric == "lpips":
        metric_instance = LPIPSMetric()
    elif metric == "dreamsim":
        metric_instance = DreamSimMetric()
    elif metric == "dino":
        metric_instance = DINODiversityMetric()
    elif metric == "dinokdd":
        metric_instance = DINOKDDMetric()
    else:
        raise ValueError("Unsupported metric type. Choose 'lpips' or 'dreamsim'.")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diversity metrics")
    parser.add_argument("--metric", type=str, choices=["lpips", "dreamsim","dino","dinokdd"], required=True, help="Choose the metric to evaluate")
    parser.add_argument("--number-of-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--label", type=int, choices=[0, 97, 300, 389, 409, 555, 569, 571, 574, 701], default=0)
    parser.add_argument("--sample-method", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--cfg", type=float, choices=[1.0, 3.0, 5.0], default=1.0)
    parser.add_argument("--sample-steps", type=int, choices=[50, 100, 150, 200, 250], default=250)
    parser.add_argument("--diffusion-form", type=str, default="sigma", choices=["sigma", "increase"])
    
    args = parser.parse_args()

    npz_path = preprocess_path(args.number_of_samples, args.label, args.sample_method, args.cfg, args.sample_steps, args.diffusion_form)
    mean, std = compute_diversity_metrics(npz_path, args.metric)

    print(f" Mean: {mean:.4f}, Std: {std:.4f}")
    