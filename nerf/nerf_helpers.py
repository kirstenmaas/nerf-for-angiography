import torch
import pandas as pd
import numpy as np
import pdb
from ast import literal_eval
import matplotlib.pyplot as plt

def map_column_to_np(df, column_name):
    # convert from strings to arrays
    column = df[column_name].apply(literal_eval)
    return np.array(column.tolist())

def randomize_depth(z_vals, device):
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape).to(device)
    z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(device)

    return depth_values

def get_minibatches(inputs, chunksize=1024*8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def get_predictions(model, flattened_query_points, chunksize, target_img_idx=None):
  batches = get_minibatches(flattened_query_points, chunksize=chunksize)

  predictions = []
  for batch in batches:
    if target_img_idx:
      indices = torch.Tensor([target_img_idx]).repeat((batch.shape[0], 1)).to(model.device)
      batch_data = torch.cat((batch, indices), dim=-1)
      predictions.append(model(batch_data))
    else:
      predictions.append(model(batch))
  
  radiance_field_flattened = torch.cat(predictions, dim=0)

  return radiance_field_flattened

def cumprod_exclusive(tensor):
  # Only works for the last dimension (dim=-1)
  dim = -1
  # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
  cumprod = torch.cumprod(tensor, dim)
  # "Roll" the elements along dimension 'dim' by 1 element.
  cumprod = torch.roll(cumprod, 1, dim)
  # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
  cumprod[..., 0] = 1.
  
  return cumprod

def render_volume_density(radiance_field, ray_directions, depth_values, raw_noise_std=0.):
  one_e_10 = torch.tensor([1e10], dtype=ray_directions.dtype, device=ray_directions.device)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

  # Multiply each distance by the norm of its corresponding direction ray
  # to convert to real world distance (accounts for non-unit directions).
  norm_dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)

  if radiance_field.shape[-1] == 2:
    sigma_a = torch.nn.functional.relu(radiance_field[..., -1])
    rgb = torch.sigmoid(radiance_field[..., :-1])
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    rgb_map = torch.squeeze((weights[..., None] * rgb).sum(dim=-2))
    depth_map = (weights * depth_values).sum(dim=-1)

    alpha_sum = torch.sum(alpha, dim=-1)
    ray_density = alpha / (alpha_sum.unsqueeze(-1) + 1e-10)
    
    ray_entropy = -torch.sum(ray_density * torch.log(ray_density + 1e-10), dim=-1)
    threshold = 0.7
    mask = (alpha_sum>threshold).detach()
    ray_entropy *= mask

    entropy = torch.mean(ray_entropy)
  else:
    # "Ensemble" with shared weights -> average
    if radiance_field.shape[-1] > 1:
      sigma_a = torch.nn.functional.relu(torch.mean(radiance_field, dim=-1))
      # sigma_a = torch.nn.Sigmoid()(torch.mean(radiance_field, dim=-1))
    else:
      sigma_a_soft = torch.nn.Softplus()(radiance_field[...,-1])
      sigma_a_sigm = torch.nn.Sigmoid()(radiance_field[..., -1])
      # print(sigma_a_relu)
      # print(sigma_a_sigm)
      
      sigma_a = sigma_a_sigm
      # clamp between 0 - 1
      # sigma_a = torch.clamp(sigma_a, min=0, max=1)
      # pdb.set_trace()
      # sigma_a = torch.nn.Softplus()(radiance_field[...,-1])

    rgb = torch.ones(sigma_a.shape[0], sigma_a.shape[1], 1).to(ray_directions.device)

    # rgb_map = torch.sum(sigma_a*norm_dists, dim=-1)#*dists)
    # invert
    # rgb_map = torch.abs(rgb_map - torch.max(rgb_map))

    alpha = torch.exp(-sigma_a * norm_dists)
    weights = (1-alpha+1e-10) * cumprod_exclusive(alpha)
    rgb_map = torch.prod(alpha, dim=-1)
    # print(torch.mean(sigma_a))

    # weights = 1-sigma_a
    # rgb_map = torch.exp(-torch.sum(sigma_a * norm_dists, dim=-1))

    # alpha = weights
    depth_map = (alpha * depth_values).sum(dim=-1)
    # pdb.set_trace()

    mask_ray_entropy = get_ray_entropy(sigma_a, rgb_map)

    entropy = mask_ray_entropy

  return rgb_map, depth_map, weights, entropy, [sigma_a, rgb]

def get_ray_entropy(sigmas, rgb_map, threshold=0.4):
  weights_sum = torch.sum(sigmas, dim=-1)
  ray_density = sigmas / (weights_sum.unsqueeze(1) + 1e-10)
  ray_entropy = -torch.sum(ray_density * torch.log(ray_density + 1e-10), dim=-1)

  per_ray_density = 1 - rgb_map
  mask = (per_ray_density>threshold).detach()

  mask_ray_entropy = ray_entropy*mask #+ entropy(1-sigma_a)*zero_mask

  return mask_ray_entropy

def sample_pixel_rays(train_ray_df, img_sample_size, device, weights=None, unseen=False):
  # sample without replacement from train ray df
  ray_batch = train_ray_df.sample(n=img_sample_size, weights=weights).sample(frac=1)
  # pdb.set_trace()

  batch_pix_vals = None
  if not unseen:
    batch_pix_vals = torch.from_numpy(ray_batch['pixel_value'].to_numpy()).to(device)
    batch_pix_vals = batch_pix_vals.float()

  batch_origins = torch.Tensor(ray_batch['ray_origins'].tolist()).to(device)
  batch_directions = torch.Tensor(ray_batch['ray_directions'].tolist()).to(device)

  return [batch_origins, batch_directions, batch_pix_vals]

def sample_image_rays(train_df, train_ray_df, img_sample_size, device, random=False):
  # sample proj from train df
  proj_batch = train_df.sample(n=1)
  proj_id = proj_batch.index[0]

  proj_ray_batch = train_ray_df[train_ray_df['image_id'] == proj_id].copy()
  
  # do not randomize the order of the pixels in the image
  if not random:
    batch_pix_vals = torch.zeros(int(np.sqrt(img_sample_size)), int(np.sqrt(img_sample_size)))
    batch_pix_vals[proj_ray_batch['x_position'].tolist(), proj_ray_batch['y_position'].tolist()] = torch.Tensor(proj_ray_batch['pixel_value'].tolist())
    
    batch_origins = torch.Tensor(proj_ray_batch['ray_origins'].tolist()).to(device)
    batch_directions = torch.Tensor(proj_ray_batch['ray_directions'].tolist()).to(device)
    batch_pix_vals = batch_pix_vals.flatten().to(device)
  # sample random pixels in image
  else:
    batch_pix_vals = torch.zeros((img_sample_size))
    batch_pix_rows = proj_ray_batch.sample(n=img_sample_size)

    batch_pix_vals = torch.Tensor(batch_pix_rows['pixel_value'].tolist()).to(device)
    batch_origins = torch.Tensor(batch_pix_rows['ray_origins'].tolist()).to(device)
    batch_directions = torch.Tensor(batch_pix_rows['ray_directions'].tolist()).to(device)
  
  return [batch_origins, batch_directions, batch_pix_vals]

def fine_sampling(depth_values, weights_coarse, ray_origins, ray_directions, coarse_model, fine_model, depth_samples_per_ray_fine, chunksize):
    pdf_depth_values = depth_values
    if len(depth_values.shape) == 1:
      pdf_depth_values = depth_values.repeat(ray_origins.shape[0], 1)
    
    depth_vals_mid = .5 * (pdf_depth_values[...,1:] + pdf_depth_values[...,:-1])
    depth_samples = sample_pdf(depth_vals_mid, weights_coarse[..., 1:-1], depth_samples_per_ray_fine, ray_origins.device)

    depth_vals, _ = torch.sort(torch.cat([pdf_depth_values, depth_samples.detach()], -1), -1)
    fine_pts = ray_origins[...,None,:] + ray_directions[...,None,:] * depth_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    flattened_fine_pts = fine_pts.reshape((-1, 3)).float()

    network = coarse_model if fine_model is None else fine_model
    radiance_field_fine = get_predictions(network, flattened_fine_pts, fine_pts.shape, chunksize, target_img_idx=None)
    rgb_map_fine, depth_map_fine, weights_fine, entropy_fine, _ = render_volume_density(radiance_field_fine, ray_directions, depth_vals)

    # pdb.set_trace()
    return rgb_map_fine, depth_map_fine, entropy_fine

def sample_pdf(bins, weights, N_samples, device):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], dim=-1) # (batch, len(bins))

    # Take uniform samples
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights)

    # Use inverse inverse transform sampling to sample the depths
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
  
def ndc_rays(sample_size, focal_length, near_thresh, batch_origins, batch_directions):
    # Shift ray origins to near plane
    t = -(near_thresh + batch_origins[..., 2]) / batch_directions[..., 2]
    batch_origins = batch_origins + t[..., None] * batch_directions

    # Projection
    o0 = -1./(sample_size/(2.*focal_length)) * batch_origins[..., 0] / batch_origins[..., 2]
    o1 = -1./(sample_size/(2.*focal_length)) * batch_origins[..., 1] / batch_origins[..., 2]
    o2 = 1. + 2. * near_thresh / batch_origins[..., 2]

    d0 = -1./(sample_size/(2.*focal_length)) * \
        (batch_directions[..., 0]/batch_directions[..., 2] - batch_origins[..., 0]/batch_origins[..., 2])
    d1 = -1./(sample_size/(2.*focal_length)) * \
        (batch_directions[..., 1]/batch_directions[..., 2] - batch_origins[..., 1]/batch_origins[..., 2])
    d2 = -2. * near_thresh / batch_origins[..., 2]

    batch_origins = torch.stack([o0, o1, o2], -1)
    batch_directions = torch.stack([d0, d1, d2], -1)

    return batch_origins, batch_directions

def sample_depth(batch_directions, depth_samples_per_ray_coarse, device):
    near_thresh = 0
    far_thresh = 1
    # near_thresh = near_thresh * torch.ones_like(batch_directions[..., :1])
    # far_thresh = far_thresh * torch.ones_like(batch_directions[..., :1])
    
    t_vals = torch.linspace(0., 1., depth_samples_per_ray_coarse).to(device)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)
    depth_values = z_vals.to(device)

    batch_depth_values = randomize_depth(depth_values, device)

    return near_thresh, far_thresh, batch_depth_values
