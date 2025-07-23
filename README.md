## Master Practical of Image and Video Synthesis  <br><sub>**Research of Diversity of SiT**</sub>
[Our Report Paper](https://arxiv.org/pdf/2401.08740.pdf)

![SiT samples](examples/odevssde—cfg/ode_sde_cfg.png)

This repository extends the original [Scalable Interpolant Transformer (SiT)](https://arxiv.org/abs/2401.08740) project to **systematically evaluate the diversity** of generated images using various perceptual and statistical metrics. Our practical focuses on **benchmarking and analyzing SiT under different sampling settings** (SDE/ODE, CFG, noise levels, etc.), and includes tools for evaluating:


---
## 📂 Folder Structure

<summary><strong>📂 Folder Structure</strong></summary>

```text
├── SiT         # Original Implementation of SiT
├── diversity_metrics/   # Implementation of Diversity Metric and Evaluation of (FID, sFID, Inception Score,Recall, Precision)   
│   ├── compute_fid.py   # Provided by our supervisor Johannes Schusterbauer
│   ├── eval.py  
│   └── metrics.py   
│   
│── exp-final/    # Experiment Results stored in this File
│   ├── 1.number_dependency/
│   ├── 2.cfg_comparison/ 
│   ├── 3.sde_comparison/
│   ├── 4.cfg_interval_study/
│   ├── tradeoff/
│   ├── metric_correlation_heatmap.png 
│   └── valloss_vs_samples.png
│        
└── validation_loss/         
        
```
---


## 🔧 Setup

Clone the repo and create the environment:

```bash
git clone https://github.com/77Han329/CVPractical.git
cd CVPractical
conda env create -f SiT/environment.yml
conda activate SiT

# Generate samples across multiple classes
torchrun --nproc_per_node=4 sample_ddp.py ODE --model SiT-XL/2 --num-fid-samples 10000

# Compute metrics from saved samples(FID,sFID,Inception Score, Precision, Recall)
python diversity_metrics/compute_fid.py \
  --ref_batch samples/VIRTUAL_imagenet256_labeled.npz \
  --sample_batch path/to/sample_batch.npz

##Visualizing Diversity
python vis.py --csv-dir --save-dir outputs/
```
---

## Experiment Results
[Download SiT Samples (CFG=1.0, ODE)](https://github.com/77Han329/CVPractical/releases/download/v1-samples/sit_samples_cfg1.0_ode_seed250.zip)

## 📦 Pre-Sampled Outputs

We provide pre-generated sample outputs of the SiT-XL/2 model for reproducibility and metric evaluation.

👉 [Download Sample Output (CFG=1.0, ODE)](https://github.com/77Han329/CVPractical/releases/download/sit-samples-v1/SiT-XL-2-pretrained-cfg-1.0-4-ODE-250-euler.npz.zip)

- Model: `SiT-XL/2`
- Config: `CFG=1.0`
- Sampler: `ODE`
- Format: `.npz`
- Size: 153 MB


## 🙏 Acknowledgements

We would like to thank our supervisor **Johannes Schusterbauer** for his invaluable support throughout the project. In particular, he provided the implementation of `compute_fid.py`, which enables comprehensive evaluation of generative diversity through FID, sFID, Inception Score, Precision, and Recall.

