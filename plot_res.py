import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


base_dir = "/home/stud/xhan/projects/cvprac/SiT/exp-local/inter-class"
sample_dirs = ["num_1000", "num_3000","num_5000","num_10000","num_20000"]
metrics = ["inception_score", "fid", "sfid", "prec", "recall"]

def extract_setting(filename):
    return os.path.splitext(filename.replace(".npz", ""))[0]


records = []
for sample_dir in sample_dirs:
    full_dir = os.path.join(base_dir, sample_dir)
    if not os.path.isdir(full_dir):
        print(f"Direction not found!!!!!{full_dir}")
        continue
    for fname in os.listdir(full_dir):
        if fname.endswith(".npz.csv"):
            fpath = os.path.join(full_dir, fname)
            try:
                df = pd.read_csv(fpath, header=0)
                row = df.iloc[0]
                records.append({
                    "sample_number": int(sample_dir.split("_")[1]),
                    "setting": extract_setting(fname),
                    "inception_score": float(row["inception_score"]),
                    "fid": float(row["fid"]),
                    "sfid": float(row["sfid"]),
                    "prec": float(row["prec"]),
                    "recall": float(row["recall"]),
                })
            except Exception as e:
                print(f"Faild to load!!!!!: {fpath}: {e}")


df_all = pd.DataFrame(records)


output_dir = os.path.join(base_dir, "plots")
os.makedirs(output_dir, exist_ok=True)


for metric in metrics:
    plt.figure(figsize=(10, 6))
    for setting in sorted(df_all["setting"].unique()):
        sub = df_all[df_all["setting"] == setting].sort_values("sample_number")
        plt.plot(sub["sample_number"], sub[metric], marker="o", label=setting)
    plt.xlabel("Sample Number")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs. Sample Number")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_vs_sample_number.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


plt.figure(figsize=(8, 6))
corr = df_all[metrics].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation between Metrics")
plt.tight_layout()

heatmap_path = os.path.join(output_dir, "metric_correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()
print(f"Saved Heatmap: {heatmap_path}")


import re


def extract_cfg(setting_str):
    match = re.search(r"cfg[_\-]?([0-9.]+)", setting_str.lower())
    return float(match.group(1)) if match else None


df_all["cfg"] = df_all["setting"].apply(extract_cfg)


if df_all["cfg"].isnull().all():
    print("No valid cfg values found in settings. Please check the setting strings.")
else:
    print("cfg values extracted successfully.")


    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for setting in sorted(df_all["setting"].unique()):
            sub = df_all[df_all["setting"] == setting]
            if sub["cfg"].notnull().any():
                plt.scatter(sub["cfg"], sub[metric], label=setting)
        plt.xlabel("CFG Scale")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs. CFG Scale")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{metric}_vs_cfg_scale.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
        
def extract_type(setting_str):
    if "sde" in setting_str.lower():
        return "SDE"
    elif "ode" in setting_str.lower():
        return "ODE"
    else:
        return "Unknown"

df_all["cfg"] = df_all["setting"].apply(extract_cfg)
df_all["sampling_type"] = df_all["setting"].apply(extract_type)


for metric in metrics:
    plt.figure(figsize=(10, 6))
    for sample_type in ["SDE", "ODE"]:
        sub = df_all[df_all["sampling_type"] == sample_type]
        grouped = sub.groupby("cfg")[metric].mean().reset_index()
        plt.plot(grouped["cfg"], grouped[metric], marker="o", label=sample_type)
    plt.xlabel("CFG Scale")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs. CFG Scale (SDE vs ODE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{metric}_cfg_sde_vs_ode.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
    

sde_df = df_all[df_all["sampling_type"] == "SDE"]

for metric in metrics:
    plt.figure(figsize=(10, 6))
    grouped = sde_df.groupby("cfg")[metric].mean().reset_index()
    plt.plot(grouped["cfg"], grouped[metric], marker="o", label="SDE")
    plt.xlabel("CFG Scale")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs. CFG Scale (SDE only)")
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{metric}_cfg_sde_only.png")
    plt.savefig(save_path)
    plt.close()
    print(f"SAved: {save_path}")