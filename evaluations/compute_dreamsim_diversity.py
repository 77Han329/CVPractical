import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dreamsim.models import DreamSimModel
from dreamsim.utils import load_model_weights


def load_images(npz_path, max_samples=100):
    print(f"Loading images from: {npz_path}")
    data = np.load(npz_path)
    if "arr_0" not in data:
        raise KeyError(f"No 'arr_0' key found in {npz_path}")
    samples = data["arr_0"][:max_samples]  # [N, H, W, C]
    tensor = torch.tensor(samples).permute(0, 3, 1, 2).float() / 255.0
    return tensor

def compute_dreamsim_diversity(images, model):
    print("Computing DreamSim distance...")
    scores = []
    with torch.no_grad():
        for i in tqdm(range(len(images))):
            for j in range(i + 1, len(images)):
                sim = model(images[i:i+1], images[j:j+1])  # similarity in [0, 1]
                distance = 1 - sim.item()  # turn into "diversity"
                scores.append(distance)
    return np.mean(scores)

def compare_selected(npz_files, max_samples=100):
    model = DreamSimModel(backbone='vit_huge_patch14')
    model = load_model_weights(model, 'vit_huge_patch14')
    model = model.eval().cuda()  # 如果你有 GPU

    scores = {}
    for name, path in npz_files.items():
        try:
            images = load_images(path, max_samples)
            if torch.cuda.is_available():
                images = images.cuda()
            score = compute_dreamsim_diversity(images, model)
            scores[name] = score
            print(f"✔️ {name}: {score:.4f}")
        except Exception as e:
            print(f"⚠️ Skipped {name} due to error: {e}")
    return scores

def plot_scores(scores):
    plt.figure(figsize=(8, 5))
    keys = list(scores.keys())
    values = list(scores.values())
    plt.bar(keys, values, color=["#5DADE2" if "ODE" in k else "#E59866" for k in keys])
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.ylabel("Avg DreamSim Diversity")
    plt.title("ODE vs SDE Diversity (DreamSim)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

def compute_average_diversity_by_method(scores):
    ode_scores = [v for k, v in scores.items() if k.startswith("ODE-label")]
    sde_scores = [v for k, v in scores.items() if k.startswith("SDE-label")]
    avg_ode = np.mean(ode_scores)
    avg_sde = np.mean(sde_scores)
    print(f"\n📊 Average DreamSim Diversity")
    print(f"ODE average: {avg_ode:.4f}")
    print(f"SDE average: {avg_sde:.4f}")
    return avg_ode, avg_sde

if __name__ == "__main__":
    base = os.path.expanduser("~/projects/cvprac/SiT/samples/ode_vs_sde")
    npz_files = {}

    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cfgs = [1.0, 2.0, 3.0, 4.0, 5.0]

    for label in labels:
        for cfg in cfgs:
            key = f"ODE-label_{label}_{cfg}"
            path = os.path.join(base, f"label_{label}/ode_cfg_{cfg}/SiT-XL-2-pretrained-cfg-{cfg}-4-ODE-250-dopri5.npz")
            npz_files[key] = path

    for label in labels:
        for cfg in cfgs:
            key = f"SDE-label_{label}_{cfg}"
            path = os.path.join(base, f"label_{label}/sde_cfg_{cfg}/SiT-XL-2-pretrained-cfg-{cfg}-4-SDE-250-Euler-sigma-Mean-0.04.npz")
            npz_files[key] = path

    scores = compare_selected(npz_files, max_samples=100)

    print("\n=== Final DreamSim Scores ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    plot_scores(scores)
    avg_ode, avg_sde = compute_average_diversity_by_method(scores)