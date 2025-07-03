import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import os
import re


csv_path = "/home/stud/xhan/projects/cvprac/SiT/fid_res.csv"
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


os.makedirs("plot", exist_ok=True)

# ==========================
# x = num_samples，y = metric
# ==========================
cfg_values = sorted(df['cfg'].dropna().unique())
colors = cm.get_cmap('tab10', len(cfg_values))
cfg_color_map = {cfg: colors(i) for i, cfg in enumerate(cfg_values)}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    print(f"\n{metric.upper()} (vs Number of Samples)")

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
            x = x[sort_idx]
            y = y[sort_idx]
            yerr = yerr[sort_idx]

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
    plt.savefig(f"plot/{metric}_vs_samples.png")
    plt.close()

# ==========================
# x = cfg，y = metric
# ==========================
for metric in metrics:
    plt.figure(figsize=(8, 5))
    print(f"\n{metric.upper()} (vs CFG)")

    for method in ['ODE', 'SDE']:
        filtered = summary[summary['method'] == method]
        avg_by_cfg = filtered.groupby('cfg')[[f"{metric}_mean", f"{metric}_std"]].mean().reset_index()

        x = avg_by_cfg['cfg'].to_numpy()
        y = avg_by_cfg[f"{metric}_mean"].to_numpy()
        yerr = avg_by_cfg[f"{metric}_std"].to_numpy()

        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        yerr = yerr[sort_idx]

        linestyle = '--' if method == 'SDE' else '-'
        plt.errorbar(x, y, yerr=yerr, fmt='.', capsize=4, linewidth=2,
                     linestyle=linestyle, label=method)

    plt.title(f"{metric.upper()} vs CFG")
    plt.xlabel("CFG")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot/{metric}_vs_cfg.png")
    plt.close()

# ==========================
# x = num_samples，y = average(metric) across all cfg+method
# ==========================

for metric in metrics:
    plt.figure(figsize=(8, 5))
    print(f"\n{metric.upper()} (vs Number of Samples, Averaged over cfg & method)")

    avg_by_sample = summary.groupby('num_samples')[[f"{metric}_mean", f"{metric}_std"]].mean().reset_index()

    x = avg_by_sample['num_samples'].to_numpy()
    y = avg_by_sample[f"{metric}_mean"].to_numpy()
    yerr = avg_by_sample[f"{metric}_std"].to_numpy()

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    yerr = yerr[sort_idx]

    visual_std_scale = 2.0
    yerr *= visual_std_scale

    plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, linewidth=2, color='black', label='Average')

    plt.title(f"{metric.upper()} (Averaged) vs Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot/{metric}_vs_samples_avg.png")
    plt.close()