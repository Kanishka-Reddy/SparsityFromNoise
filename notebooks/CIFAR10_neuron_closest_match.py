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

import matplotlib.pyplot as plt
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('../py_scripts')
from py_scripts import LightningDataModule, get_params_net_dataloader
from py_scripts import Vanilla_Dataset, calculate_fid_given_torch_datasets, calculate_means_and_covs_given_torch_datasets, calculate_frechet_distance, calculate_single_mean_and_cov_given_torch_dataset
import glob
import pickle
import torch.nn.functional as F
from diffusion_utils import *

# DONT NEED TO USE GPU HERE

use_gpu = True 

if use_gpu: 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: 
    device="cpu"



#%%
#LowerLR
model_template = "Interpretable_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"
#"_ffnCIFAR10_w_Projections_Adam_lr0.0001_datas=None_10000Neurons_projM=True_nlayers1"
#"_ffnRaw_CIFAR10_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1" #"_ffnRaw_CIFAR10_Adam_lr1e-05_datas=None_10000Neurons_projM=False_nlayers1"#"_ffnBaseline_Adam_lr0.0001_datas=None_1steps_10000Neurons_projM=False_nlayers1"

use_CIFAR10 = True 

compute_fid = True

bs=10

model_prefixes = [0.0,0.05,0.1,0.3,0.8,1.5, 3.0, 10.0]
#["100N","1_000N","10_000N","100_000N"]

dataset_path="../data/"
save_dir = "../../scratch_link/Foundational-SDM/data/CachedLatents/"
extra = {"use_wandb":False}

if "CIFAR10" in model_template or use_CIFAR10: 
    latent, labels = torch.load('../data/CIFAR10/all_data_train.pt')
    latent = latent.flatten(start_dim=1 )
    if latent.dtype is torch.uint8:#"/ImageNet32/" in self.dataset_path or "/CIFAR10/" in self.dataset_path:
            latent = latent.type(torch.float)/255
else: 

    latent, labels = torch.load('../data/CachedOutputs/ConvMixerWTransforms_ImgNet32_CIFAR10/all_data_train.pt')

latent = latent.to(device )


if compute_fid: 
    dims = 2048
    batch_size = 256
    num_workers = 0
    m1,s1 = calculate_single_mean_and_cov_given_torch_dataset( Vanilla_Dataset(latent.view(len(latent), 3, 32, 32)), batch_size , device, dims, num_workers)

li_mean_max_cosine_sim = []
li_weighted_mean_max_cosine_sim = []
li_fid = []
for run_ind, run in enumerate(model_prefixes): 
    print("Noise amount:", run)

    model, params = load_model(f"{run}{model_template}", dataset_path, save_dir, device, extra_extras=extra)

    if run_ind ==0:
        print(model)

    #######

    rand_inds = np.random.choice(active_inds, 50)
    ws = min_max_scale( model.X_a.weight.detach()[rand_inds] )
    gridshow( ws.view(50, 3,32,32), title=f"Keys | Random Neurons | Noise amount={run}", nimages=50 )


    # rows are neurons. 
    dists = cosine_sim_matrices(model.X_a.weight.detach(), latent)  
    #dists = torch.cdist(model.X_a.weight.detach(), latent/torch.norm(latent,dim=1, keepdim=True), p=2.0)

    dist_vals, dist_inds = dists.max(dim=1) #dists.min(dim=1) #dists.max(dim=1)
    neuron_active_summer = get_active_neurons(model, latent, device, params.nneurons[0])
    active_mask = neuron_active_summer>0.0001

    mean_max_cosine_sim = dist_vals.mean()
    weighted_mean_max_cosine_sim = (dist_vals* (neuron_active_summer/neuron_active_summer.sum())).sum()

    plt.scatter(neuron_active_summer.cpu(), dist_vals.cpu())
    plt.xlabel("Neuron activity")
    plt.ylabel("Max cosine similarity")

    plt.title("Max cosine sim as a function of neuron activity")
    plt.show()

    print("weighted mean max cosine sim", weighted_mean_max_cosine_sim,"| mean max cosine sim", mean_max_cosine_sim)

    li_mean_max_cosine_sim.append(mean_max_cosine_sim.cpu())
    li_weighted_mean_max_cosine_sim.append(weighted_mean_max_cosine_sim.cpu())

    print("Fraction of alive neurons", active_mask.type(torch.float).mean())

    kvals, kinds = torch.topk(dist_vals, bs)
    print(f"Top {bs} cosine sims", kvals)
    print("Most active neuron is", neuron_active_summer.max())
    print(f"How active each of these neurons is:", neuron_active_summer[kinds])
    print(f"Activity as a percentage:", neuron_active_summer[kinds]/neuron_active_summer.sum())
    ws = min_max_scale( model.X_a.weight.detach()[kinds] )
    gridshow( ws.view(bs, 3,32,32), title=f"Keys | Highest Cosine Similarity Neuron | Noise amount={run}" )
    gridshow( latent[dist_inds[kinds]].view(bs, 3,32,32), title=f"Closest Images | Noise amount={run}" )

    ws = min_max_scale( model.X_v().detach()[kinds] )
    gridshow( ws.view(bs, 3,32,32), title=f"Values | Highest Cosine Similarity Neuron | Noise amount={run}" )

    print("----------")

    active_inds = torch.arange(params.nneurons[0])[active_mask]
    rand_inds = np.random.choice(active_inds, bs)
    ws = min_max_scale( model.X_a.weight.detach()[rand_inds] )
    gridshow( ws.view(bs, 3,32,32), title=f"Random Active Neurons Keys | Noise amount={run}" )

    gridshow( latent[dist_inds[rand_inds]].view(bs, 3,32,32), title=f"Closest Images | Noise amount={run}" )

    print("----------")

    _, act_inds = torch.topk(neuron_active_summer, bs)
    ws = min_max_scale( model.X_a.weight.detach()[act_inds] )
    gridshow( ws.view(bs, 3,32,32), title=f"Most Active Neurons Keys | Noise amount={run}" )

    gridshow( latent[dist_inds[act_inds]].view(bs, 3,32,32), title=f"Closest Images | Noise amount={run}" )

    ws = min_max_scale( model.X_v().detach()[act_inds] )
    gridshow( ws.view(bs, 3,32,32), title=f"Values | Most Active Neurons | Noise amount={run}" )

    print("----------")

    _, top_most_act_inds = torch.topk(dists[act_inds[0]], bs)
    gridshow( latent[top_most_act_inds].view(bs, 3,32,32), title=f"Closest Images for most active neuron | Noise amount={run}" )

    if compute_fid:
        print("Computing FID")
        m2,s2 = calculate_single_mean_and_cov_given_torch_dataset( Vanilla_Dataset(model.X_a.weight.detach().view(params.nneurons[0],3,32,32)), batch_size , device, dims, num_workers)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        li_fid.append(fid_value)
        print("FID is:", fid_value)

plt.scatter(model_prefixes, li_mean_max_cosine_sim, label="uniform")
plt.scatter(model_prefixes, li_weighted_mean_max_cosine_sim, label="weighted")
plt.xlabel("Diff Noise")
plt.ylabel("Cosine similarity")
plt.legend()
plt.title("Average Max Cosine similarity for each neuron as a function of diffusion noise")
plt.show()

plt.scatter(model_prefixes, li_fid, label="FID Scores")
plt.xlabel("Diff Noise")
plt.ylabel("FID Score")
plt.legend()
plt.title("FID of Neurons Receptive fields as a function of diffusion noise (lower is better)")
plt.show()

#%%
li_fid
#%%
model_prefixes
#%%
