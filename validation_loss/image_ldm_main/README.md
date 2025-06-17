# Image Latent Diffusion Model


## 🚀 Usage

### Setup New Project

For a new project you need to implement a `LightningModule` and specify the train logics and parameters in the `config.yaml`.


### Training

#### On GPU

You can run experiments by creating a config file, similar to the ones already there. To then start the training just run

```bash
python train.py
```

which will launch the training with the default `config.yaml`. If you prefer to train your models using `accelerate`, you can use the corresponding `quick_train.py` function, which reads parameters from the same config. In some cases, the `accelerate` script might speed up training (e.g. 9.4 vs 6.9 it/s for `bf16` training with dummy data). See below under *Setup* how to setup a accelerate compatible lightning module.

You can enable wandb logging with `use_wandb=True` or offline wandb logging with `use_wandb_offline=True`. By default it will log to tensorboard.

To start a specific experiment just run `experiment=folder/<exp-file>.yaml`. You can overwrite specific parameters with `key=val`, e.g. `data=lhq` to change the dataset or `name=myname/myexp` to change the name of the experiment. If you want to add a parameter to a config, that has not been there in the original config, just add a plus sign before, e.g. `+trainer_module.params.new_param=2`.

Use `--info config` to only show the resulting config.

#### Slurm-Cluster

You can use the scripts provided in `scripts/slurm/` to start slurm jobs.

```bash
# MVL
bash scripts/slurm/start_mvl.sh -n <slurm-name> -t 02:00:00 --gpus 1 --partition a100 --args experiment=... name=...

# Juelich
bash scripts/slurm/start_juelich.sh -n <slurm-name> -t 02:00:00 --nodes 1 --partition booster --args experiment=... name=...
```

The only difference between both is, that on MVL you specify the number of GPUs (`--gpus`) and on Juelich you specify the number of nodes (`--nodes`). After `--args` you can pass all arguments for hydra. The scripts automatically set the  `tqdm` refresh rate to 50, to avoid super large slurm log files.

You can also add a dependency on a previous run. For that you must already know the checkpoint folder to specify the `last.ckpt` and the SLURM ID. You can add a dependency by inserting the line `--dependency=afterany:<id>` in the starting script (e.g. `scripts/slurm/start_mvl.sh`), and then of course, adding the `resume_checkpoint=your/path/checkpoints/last.ckpt` or `resume_checkpoint=your/path/checkpoints/interrupted.ckpt`.

### Resuming

If you want to resume from a checkpoint just use
```bash
python train.py experiment=<your-exp> resume_checkpoint=<path>.ckpt
```
If you only want to load the weights from a checkpoint without resuming the training state, you can use `load_weights=<path>.ckpt`.


### Profiling

You can enable profiling by setting `profile=true`. This will save a json file `profile.json` which you can load into [this](https://ui.perfetto.dev/) web UI. In `scripts/profiling` you find additional scripts to profile your model.


## 🛠️ Setup


### install

```bash
# create conda environment
conda env create -f environment.yml

# manually install jutils
pip install git+https://github.com/joh-schb/jutils.git#egg=jutils
```

### Model checkpoints

Make sure to have a folder in this directory which includes all the neccessary checkpoints, e.g. the autoencoder checkpoint for the first stage model.

Just link the whole checkpoint folder, which includes all checkpoints.

*Juelich:*
```bash
ln -s /p/scratch/degeai/fischer13/checkpoints checkpoints
```

*Heidelberg:*
```bash
ln -s /export/compvis-nfs/user/jfischer/checkpoints checkpoints
```

*MVL:*
```bash
ln -s /export/scratch/ru59wap/checkpoints checkpoints
```

Or you can freshly create this folder by downloading all individual checkpoints:
```bash
mkdir checkpoints
cd checkpoints

# SD checkpoint
wget -O v2-1_768-ema-pruned.ckpt https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt?download=true

# SD Autoencoder checkpoint
wget -O sd_ae.ckpt https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=0

# TinyAutoencoderKL checkpoints
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_encoder.pth
wget https://github.com/madebyollin/taesd/raw/refs/heads/main/taesd_decoder.pth

# SiT-XL-2-256 checkpoint
wget -O SiT-XL-2-256.pt https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0

# nulltext openclip embedding
wget -O nulltext_openclip_embedding.npy https://www.dropbox.com/scl/fi/6yq31ho8vntz7bbvu3ad2/nulltext_openclip_embedding.npy?rlkey=gcy6vtdg61u6fdhavvxsixebx&st=xf9gxhdz&dl=0
```

### Lightning Module w. Accelerate compatibility

Here are some rules to setup the `LightningModule`, s.t. it can seamlessly be used with the provided accelerate `quick_train.py` training script. The following functions are called by the accelerate script:

- **`configure_optimizers`** (see example in `trainer.py`)
- **`forward` function of `LightningModule`**
    - Return either `loss` or a tuple `(loss, loss_dict)`.
    - No `self.log(...)` calls, these should go into `training_step`.
    - No `log_grad_norm`, this should go into `training_step`.
- **`on_train_batch_end`**
    - EMA updates and lr-schedulers steps should go in here.
    - LR scheduling is handled via the lightning trainer or the accelerate script. Hence, for the lr-scheduler first check if `self._trainer` exists, s.t. during optimization w. accelerate it is ignored.
    - Should contain `stop_training_method`.
- **`validation_step`**
    - No `self.log(...)` calls.
- **`on_validation_epoch_end`**
    - Can contain `self.log(...)` calls, but currently only for single values (no dictionaries).
    - Metric aggregation should be handled via `torchmetrics` (handles aggreation automatically) or manually by using `torch.distributed` (rather not recommended).
    - For image or video logging, use dedicated functions implemented in `ldm/logging.py` that get the `logger` instance as an argument (e.g. `log_images(self.logger, ims, ...)`).

 You can find a general example in `trainer.py`. Checkpoints are saved, so that they can be mutually loaded. If you encounter any issues, feel free to solve them.😉
