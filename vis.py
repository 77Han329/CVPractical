import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
import argparse


# ======= Argument Parser =======
parser = argparse.ArgumentParser(description="Visualize metrics from .csv")
parser.add_argument("--input", type=str, default="fid_res.csv", help="Path to input CSV file")
parser.add_argument("--outdir", type=str, default="plot", help="Directory to save plots")
args = parser.parse_args()

csv_path = args.input
out_dir = args.outdir
os.makedirs(out_dir, exist_ok=True)

# ======= Read and parse data =======
df = pd.read_csv(csv_path)

def parse_info(path):
    cfg_match = re.search(r'cfg_([0-9.]+)', path)
    method_match = re.search(r'(ODE|SDE)', path)
    seed_match = re.search(r'seed_(\d+)', path)
    num_match = re.search(r'num_samples_(\d+)', path)
    cfg = float(cfg_match.group(1)) if cfg_match else None
    method = method_match.group(1) if method_match else None
    seed = int(seed_match.group(1)) if seed_match else None
    num_samples = int(num_match.group(1)) if num_match else None
    return cfg, method, seed, num_samples

df[['cfg', 'method', 'seed', 'num_samples']] = df['sample_batch'].apply(
    lambda x: pd.Series(parse_info(x))
)

metrics = ['fid', 'sfid', 'inception_score', 'precision', 'recall']

summary = df.groupby(['cfg', 'method', 'num_samples'])[metrics].agg(['mean', 'std']).reset_index()
summary.columns = ['cfg', 'method', 'num_samples'] + [f"{m}_{stat}" for m in metrics for stat in ['mean', 'std']]

cfg_values = sorted(df['cfg'].dropna().unique())
colors = cm.get_cmap('tab10', len(cfg_values))
cfg_color_map = {cfg: colors(i) for i, cfg in enumerate(cfg_values)}

# ======= Plot 1: metric vs num_samples per cfg & method =======
for metric in metrics:
    plt.figure(figsize=(10, 6))
    print(f"\n{metric.upper()} vs Number of Samples")

    for cfg in cfg_values:
        color = cfg_color_map[cfg]
        for method in ['ODE', 'SDE']:
            group = summary[(summary['cfg'] == cfg) & (summary['method'] == method)]
            if group.empty:
                continue
            x = group['num_samples'].to_numpy()
            y = group[f"{metric}_mean"].to_numpy()
            yerr = group[f"{metric}_std"].to_numpy()
            sort_idx = np.argsort(x)
            x, y, yerr = x[sort_idx], y[sort_idx], yerr[sort_idx]
            linestyle = '--' if method == 'SDE' else '-'
            label = f"cfg={cfg} {method}"

            plt.errorbar(x, y, yerr=yerr, fmt='.', capsize=4, linewidth=2,
                         linestyle=linestyle, color=color, label=label)

    plt.title(f"{metric.upper()} vs Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{metric}_vs_samples.png")
    plt.close()

# ======= Plot 2: metric vs cfg (averaged over num_samples) =======
for metric in metrics:
    plt.figure(figsize=(8, 5))
    print(f"\n{metric.upper()} vs CFG")

    for method in ['ODE', 'SDE']:
        filtered = summary[summary['method'] == method]
        avg_by_cfg = filtered.groupby('cfg')[[f"{metric}_mean", f"{metric}_std"]].mean().reset_index()
        x = avg_by_cfg['cfg'].to_numpy()
        y = avg_by_cfg[f"{metric}_mean"].to_numpy()
        yerr = avg_by_cfg[f"{metric}_std"].to_numpy()
        sort_idx = np.argsort(x)
        x, y, yerr = x[sort_idx], y[sort_idx], yerr[sort_idx]
        linestyle = '--' if method == 'SDE' else '-'

        plt.errorbar(x, y, yerr=yerr, fmt='.', capsize=4, linewidth=2,
                     linestyle=linestyle, label=method)

    plt.title(f"{metric.upper()} vs CFG")
    plt.xlabel("CFG")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{metric}_vs_cfg.png")
    plt.close()

# ======= Plot 3: Averaged metric vs num_samples (across cfg+method) =======
for metric in metrics:
    plt.figure(figsize=(8, 5))
    print(f"\n{metric.upper()} Averaged vs Number of Samples")

    avg_by_sample = summary.groupby('num_samples')[[f"{metric}_mean", f"{metric}_std"]].mean().reset_index()
    x = avg_by_sample['num_samples'].to_numpy()
    y = avg_by_sample[f"{metric}_mean"].to_numpy()
    yerr = avg_by_sample[f"{metric}_std"].to_numpy()
    sort_idx = np.argsort(x)
    x, y, yerr = x[sort_idx], y[sort_idx], yerr[sort_idx]
    yerr *= 2.0  # visually enlarge std

    plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, linewidth=2,
                 color='black', label='Average')

    plt.title(f"{metric.upper()} (Averaged) vs Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{metric}_vs_samples_avg.png")
    plt.close()