import os
import sys
import torch
import inspect
from omegaconf import OmegaConf
from jutils import instantiate_from_config

currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def main():

    paths = [
        "configs/model/dit-b-2.yaml",
        "configs/model/dit-l-2.yaml",
        "configs/model/dit-xl-2.yaml",
        "configs/model/sd-v21.yaml"
    ]

    for cfg_path in paths:
        cfg = OmegaConf.load(cfg_path)
        unet = instantiate_from_config(cfg)
        n_params = sum(p.numel() for p in unet.parameters())
        name = os.path.basename(cfg_path).split(".")[0]
        print(f"{name:<16}: {n_params:,}")

    # dev = "cuda:7"
    # unet.to(dev)
    # x = torch.randn((32, 3, 128, 128)).to(dev)
    # print(unet.encode(x).shape)
    # print(unet.decode(x).shape)


if __name__ == "__main__":
    main()
