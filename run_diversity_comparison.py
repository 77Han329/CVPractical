import os
import numpy as np
import torch
import lpips
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_images(npz_path, max_samples=100):
    print(f"Loading Images from: {npz_path}")
    samples = np.load(npz_path)["arr_0"]
    samples = samples[:max_samples]
    samples = torch.tensor(samples).permute(0, 3, 1, 2).float() / 127.5 - 1  # [0, 255] → [-1, 1]
    return samples

def compute_lpips_diversity(images):
    print("Computing LPIPS Diversity...")
    model = lpips.LPIPS(net='alex').cuda()
    scores = []
    for i in tqdm(range(len(images))):
        for j in range(i + 1, len(images)):
            d = model(images[i:i+1].cuda(), images[j:j+1].cuda())
            scores.append(d.item())
    return np.mean(scores)

def compare_lpips(npz_paths, max_samples=100):
    results_lpips = {}
    for name, path in npz_paths.items():
        images = load_images(path, max_samples)
        results_lpips[name] = compute_lpips_diversity(images)
    return results_lpips

def plot_results(title, results, ylabel):
    plt.bar(results.keys(), results.values(), color=["steelblue", "tomato"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    npz_paths = {
        "ODE": os.path.expanduser("~/projects/cvprac/SiT/samples/SiT-XL-2-pretrained-cfg-1.0-4-ODE-250-dopri5.npz"),
        "SDE": os.path.expanduser("~/projects/cvprac/SiT/samples/SiT-XL-2-pretrained-cfg-1.0-4-SDE-250-Euler-sigma-Mean-0.04.npz")
    }

    lpips_scores = compare_lpips(npz_paths, max_samples=100)

    print("\n LPIPS Diversity")
    for name, score in lpips_scores.items():
        print(f"{name}: {score:.4f}")

    plot_results("LPIPS Diversity: ODE vs SDE", lpips_scores, "Average LPIPS Score")