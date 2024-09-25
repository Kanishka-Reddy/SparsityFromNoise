#%%


%load_ext autoreload
%autoreload 2
%matplotlib inline
import torch 
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd 
import json 
import copy 
import pickle 
import os 
import sys
import copy
import umap
import wandb
from pytorch_lightning.loggers import WandbLogger
import seaborn as sns

import matplotlib.pyplot as plt
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('../py_scripts')
from py_scripts import LightningDataModule, get_params_net_dataloader
import glob
import pickle
import torch.nn.functional as F
from diffusion_utils import *

# DONT NEED TO USE GPU HERE

use_gpu = False

if use_gpu: 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: 
    device="cpu"


#%%

train_o_test = "train"
use_CIFAR10 = True

for use_CIFAR10 in [True, False]:

    if use_CIFAR10: 
        latent, labels = torch.load(f'../data/CIFAR10/all_data_{train_o_test}.pt')
        latent = latent.flatten(start_dim=1 )
        if latent.dtype is torch.uint8:#"/ImageNet32/" in self.dataset_path or "/CIFAR10/" in self.dataset_path:
                latent = latent.type(torch.float)/255
    else: 

        latent, labels = torch.load(f'../data/CachedOutputs/ConvMixerWTransforms_ImgNet32_CIFAR10/all_data_{train_o_test}.pt')

    var = torch.norm(latent, dim=1)
    print(var.shape, latent.shape )
    plt.hist(var, bins=50 )
    plt.title(f"CIFAR10={use_CIFAR10} | L2 norms")
    plt.show()
#%%
for n in [0.0, 0.3, 0.8, 1.5, 10.0]:
    noisy = latent + n*torch.randn(latent.shape)
    var = torch.norm(noisy, dim=1)
    plt.hist(var, bins=50 )
    plt.title(f"CIFAR10={use_CIFAR10} | L2 norms | Noise {n}")
    plt.show()
#%%
