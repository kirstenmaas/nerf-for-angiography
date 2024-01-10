import torch
import pandas as pd
import numpy as np
import pdb
from ast import literal_eval
import matplotlib.pyplot as plt
import nerfacc
from torch_scatter import scatter_mul

def acc_ray_marching(radiance_field, ray_origins, ray_directions, depth_samples_per_ray, near_thresh, far_thresh, early_stop_eps=1e-2, alpha_thre=1e-3):
  def alpha_fn(t_starts, t_ends, ray_indices):
    
    t_origins = ray_origins[ray_indices]  # (n_samples, 3)\
    t_dirs = ray_directions[ray_indices]  # (n_samples, 3)\
    positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0

    # n_samples is different for each ray
    # dists = torch.cat([(t_ends[torch.argwhere(ray_indices == i)] - t_starts[torch.argwhere(ray_indices == i)]) for i in range(ray_origins.shape[0])], dim=-1).squeeze().T #(n_rays, n_samples_per_ray)\
    dists_flat = (t_ends - t_starts) # (n_samples, 1)\

    predictions = radiance_field(positions)
    sigmas = torch.nn.Sigmoid()(predictions)
    alphas = torch.exp(-sigmas * dists_flat)
    
    return 1-alphas  # (n_samples, 1)\

  render_step_size = (far_thresh - near_thresh) / depth_samples_per_ray
  # pdb.set_trace()
  ray_indices, t_starts, t_ends = nerfacc.ray_marching(ray_origins, ray_directions, alpha_fn=alpha_fn, near_plane=near_thresh, far_plane=far_thresh, early_stop_eps=early_stop_eps, alpha_thre=alpha_thre, render_step_size=render_step_size)

  return ray_indices, t_starts, t_ends

def acc_render_volume_density(predictions, ray_indices, t_starts, t_ends, n_rays, depth_samples_per_ray):
  dists_flat = (t_ends - t_starts)# (n_samples, 1)\

  sigmas = torch.nn.Sigmoid()(predictions)
  alphas = torch.exp(-sigmas * dists_flat)

  index = ray_indices[:, None].expand(-1, alphas.shape[-1]).type(torch.int64)
  outputs = torch.ones((n_rays, alphas.shape[-1]), device=alphas.device, dtype=alphas.dtype)
  
  rgb_map = scatter_mul(alphas, index, dim=0, out=outputs).squeeze().float()

  return rgb_map
