from sre_constants import NOT_LITERAL
import pandas as pd
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('.')
from torch.utils.tensorboard import SummaryWriter
from ast import literal_eval
from datetime import datetime
import pdb
from nerfacc import OccupancyGrid, ContractionType
import pyvista as pv
import argparse
import ast
import time

from nerf_helpers import randomize_depth, get_predictions, render_volume_density, sample_pixel_rays, sample_image_rays
from nerf_helpers_acc import acc_ray_marching, acc_render_volume_density, acc_update_n_step
from model.CPPN import CPPN

torch.set_printoptions(precision=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--limited_size', help='Angle range to sample the projections in')
parser.add_argument('--number_angles', help='Number of projections to sample per axis')
parser.add_argument('--center_point', help='Center point for the angle sampling')
parser.add_argument('--binary', help='Whether images are binary or not')
parser.add_argument('--sampling_strategy', help='What sampling strategy to use, options: frangi, segmentation or random')
parser.add_argument('--data_name', help='Either CT data or LCA data')
parser.add_argument('--num_layers', help='Number of layers for MLP')
parser.add_argument('--num_hidden_units', help='Number of hidden units for MLP')

args = parser.parse_args()
arg_dict = vars(args)

# define general parameters
limited_size = float(arg_dict['limited_size']) if arg_dict['limited_size'] is not None else float(180)
number_angles = float(arg_dict['number_angles']) if arg_dict['number_angles'] is not None else float(4) #(+1) e.g. 5 + 1 = 6
center_point = ast.literal_eval(arg_dict['center_point']) if arg_dict['center_point'] is not None else [90, 0] #[0,90] #[90,0]
binary = arg_dict['binary'] == 'True' if arg_dict['binary'] is not None else False
sampling_strategy = arg_dict['sampling_strategy'] if arg_dict['sampling_strategy'] is not None else 'frangi'
data_name = arg_dict['data_name'] if arg_dict['data_name'] else 'ct'
num_layers = int(arg_dict['num_layers']) if arg_dict['num_layers'] else 4
num_hidden_units = int(arg_dict['num_hidden_units']) if arg_dict['num_hidden_units'] else 128

categories = ['Background']
if binary:
    categories = ['Sparse projections', 'Limited projections']

if num_hidden_units != 128 or num_layers != 4:
    categories = ['Model architecture']

cathlab = True
unseen = False

data_size = 100 if data_name == 'ct' else 162

if number_angles > 0:
    step_size = limited_size/number_angles
else:
    step_size = limited_size

outside = 100

early_stop_eps=1e-2 #1e-2
alpha_thre=1e-4
vessel_alpha_thre=5e-2

if cathlab:
    file_name = f'clinical-angles'
else:
    # file_name = f'sparse-{number_angles}'
    file_name = f'background-{limited_size}-{number_angles}-{center_point}'
    if binary:
        file_name = f'limited-sparse-{limited_size}-{number_angles}-{center_point}'
    
    

proj_df, ray_df, store_folder_name, unseen_ray_df = load_data(data_name, file_name, unseen, binary, data_size, step_size)

# create test df
test_proj_id = proj_df.index[-1] #test_proj_row.index[0]
test_ray_df = ray_df[ray_df['image_id'] == test_proj_id].copy()

test_origins = torch.from_numpy(np.array([test_ray_df['ray_origins_x'].tolist(), test_ray_df['ray_origins_y'].tolist(), test_ray_df['ray_origins_z'].tolist()]).T).float().to(device)
test_directions = torch.from_numpy(np.array([test_ray_df['ray_directions_x'].tolist(), test_ray_df['ray_directions_y'].tolist(), test_ray_df['ray_directions_z'].tolist()]).T).float().to(device)

# setup test proj from rays
test_x_positions = list(test_ray_df['x_position'].to_numpy().astype('int'))
test_y_positions = list(test_ray_df['y_position'].to_numpy().astype('int'))

img_width = int(np.max(test_x_positions)+1)
img_height = int(np.max(test_y_positions)+1)
test_img = torch.zeros((img_width, img_height)).to(device)
test_pix_vals = torch.from_numpy(test_ray_df['pixel_value'].to_numpy()).to(device).float()
test_img[test_x_positions, test_y_positions] = test_pix_vals

# when binary == False, save pixel values with weight to save the best model based only on those pixels
test_dist_df = test_ray_df[test_ray_df['distance_pixel_value'] > test_ray_df['distance_pixel_value'].mean()]
test_dist_x_pos = test_dist_df['x_position'].to_numpy().astype('int')
test_dist_y_pos = test_dist_df['y_position'].to_numpy().astype('int')
test_img_vessel = test_img[test_dist_x_pos, test_dist_y_pos]

coarse_test_pred_img = torch.zeros((img_width, img_height)).to(device)

print('loading dataframes...')

# create train df
# this takes some time as we need to map the lists to numpy arrays for the whole df
train_df = proj_df.copy()
train_ray_df = ray_df.copy()

train_ray_df['ray_origins'] = np.array([train_ray_df['ray_origins_x'].tolist(), train_ray_df['ray_origins_y'].tolist(), train_ray_df['ray_origins_z'].tolist()]).T.tolist()
train_ray_df['ray_directions'] = np.array([train_ray_df['ray_directions_x'].tolist(), train_ray_df['ray_directions_y'].tolist(), train_ray_df['ray_directions_z'].tolist()]).T.tolist()

# precomputed params for projs
focal_length = proj_df['focal_length'][0]
near_thresh = proj_df['near_thresh'][0]
far_thresh = proj_df['far_thresh'][0]
depth_samples = proj_df['depth_sample'][0]
src_pt_z = proj_df['src_pt_z'][0]

grid_scaling_factor = 1

# precompute depth values
depth_samples_per_ray_coarse = 300

mid_thresh = src_pt_z#(far_thresh + near_thresh) / 2

near_thresh = mid_thresh - outside
far_thresh = mid_thresh + outside
print(near_thresh, far_thresh)

t_vals = torch.linspace(0., 1., depth_samples_per_ray_coarse)
z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)
depth_values = z_vals.to(device)

# learning params
n_iters = 500000
early_stop_iters = 50000
display_every = 500
save_every = display_every*100
batch_size = 1024*128 # batch_size for model predictions /65536
coarse_lr = 1e-4#1e-5
lr_decay = 500
decay_rate = 0.1
decay_steps = lr_decay * 1000

# custom run parameters
sample_mode = 'pixel' #or image
sample_size = 75 #nb rays sampled per dimension per iteration
img_sample_size = sample_size**2 #total nb rays sampled per iteration
unseen_sample_size = 0
raw_noise_std = 0 #std dev of noise added to regularize sigma_a output, 1e0 recommended

# model
start_pos_enc_basis = 0
pos_enc_basis = 5
fourier_sigma = 5

# barf alpha
barf_start = 8000
barf_stop = 250000
barf_step_size = pos_enc_basis / (barf_stop - barf_start)
params = {
    'num_early_layers': num_layers,
    'num_late_layers': 0,
    'num_filters': num_hidden_units,
    'num_input_channels': 3,
    'num_output_channels': 1,
    'num_input_channels_views': 0,
    'use_bias': True,
    'pos_enc': 'none',
    'pos_enc_basis': pos_enc_basis,
    'act_func': 'relu',
    # 'sine_weights': 15,
    'fourier_sigma': fourier_sigma,
    'num_img': 1,
    'device': device,
}

# setup model
coarse_params = dict(params)

coarse_model = CPPN(coarse_params)
print(coarse_model)

coarse_model.to(device)
coarse_model.update_barf_alpha(start_pos_enc_basis, 'pts') if coarse_model.use_pos_enc == 'barf' else None
# coarse_model.apply(init_normal)

# initialize nerfAcc grid
scene_aabb = torch.tensor([-outside, -outside, -outside, outside, outside, outside], dtype=torch.float32, device=device)
acc_grid = OccupancyGrid(roi_aabb=scene_aabb, resolution=128, contraction_type=ContractionType.AABB).to(device)
vessel_acc_grid = OccupancyGrid(roi_aabb=scene_aabb, resolution=128, contraction_type=ContractionType.AABB).to(device)

pv_grid = pv.UniformGrid()
pv_grid.dimensions = np.array(acc_grid.binary.cpu().numpy().astype('int').shape) + 1

vessel_pv_grid = pv.UniformGrid()
vessel_pv_grid.dimensions = np.array(vessel_acc_grid.binary.cpu().numpy().astype('int').shape) + 1

coarse_optimizer = torch.optim.Adam(list(coarse_model.parameters()), lr=coarse_lr)

# setup tensorboard

time_zone = datetime.now().astimezone()
exp_name = time_zone.strftime("%Y-%m-%d-%H%M")
iso_time = time_zone.isoformat()

log_dir = store_folder_name + 'runs/' + exp_name + '/'
writer = SummaryWriter(log_dir=log_dir)

layout = {
    "ABCDE": {
        "mean": ["Multiline", ["mean/train", "mean/train-pred"]],
        "loss": ["Multiline", ["loss/train", "loss/test"]],
        "psnr": ["Multiline", ["psnr/train", "psnr/test"]],
    },
}
writer.add_custom_scalars(layout)

sampling_strategies = []
if sampling_strategy == 'frangi':
    sampling_strategies.append('Frangi sampling')
elif sampling_strategy == 'segmentation':
    sampling_strategies.append('Segmentation sampling')
elif sampling_strategy == 'random':
    sampling_strategies.append('Random sampling')
sampling_strategies.append('AccNeRF')

# summarize data in dictionary
page_data = {
    'ID': exp_name,
    'Date start': iso_time,
    'Category': categories,
    'Sparse projections': int((number_angles+1)**2),
    'Limited projections': int(limited_size),
    'Translation': 'None',
    'Rotation': 'None',
    'Data': data_name.upper(),
    'Binary': binary,
    'Sampling': sampling_strategies,
    'Model architecture': f"{params['num_early_layers']}x{params['num_filters']}",
    'Positional encoding': params['pos_enc'].capitalize(),
    'Learning rate': coarse_lr,
    'Centerpoint': f'({center_point[0]} {center_point[1]})'
}

# write data df to folder
binary_str = 'binary' if binary else ''
ray_df.to_csv(log_dir + '/' + f'df-rays-{file_name}-{binary_str}-{img_height}.csv', sep=';')
proj_df.to_csv(f'{log_dir}/df-{file_name}-{binary_str}-cttoproj.csv', sep=';')

print('start training...')

highest_psnr = 0
highest_iter = 0

for n_iter in range(n_iters+1):
    start_time = time.time()
    coarse_model.train()

    # update barf_alpha
    if coarse_model.use_pos_enc == 'barf' and n_iter >= barf_start and n_iter < barf_stop:
        #pts
        curr_barf_alpha = coarse_model.barf_alpha
        new_barf_alpha = curr_barf_alpha + barf_step_size
        coarse_model.update_barf_alpha(new_barf_alpha, 'pts')

    # sample rays from pixels of ALL training images
    if sample_mode == 'pixel':
        sample_weights_name = 'distance_pixel_value'
        batch_origins, batch_directions, batch_pix_vals = sample_pixel_rays(train_ray_df, img_sample_size, device, weights=sample_weights_name)
    # sample rays per training image
    else:
        batch_origins, batch_directions, batch_pix_vals = sample_image_rays(train_df, train_ray_df, img_sample_size, device)

    # update nerf acc occupancy grid
    # and nerf acc efficient raymarching
    with torch.no_grad():
        acc_grid = acc_update_n_step(acc_grid, coarse_model, n_iter, occ_thre=alpha_thre)
        vessel_acc_grid = acc_update_n_step(vessel_acc_grid, coarse_model, n_iter, occ_thre=vessel_alpha_thre)
        ray_indices, t_starts, t_ends = acc_ray_marching(coarse_model, acc_grid, scene_aabb, batch_origins, batch_directions, depth_samples_per_ray_coarse, near_thresh, far_thresh, early_stop_eps, alpha_thre)

    if len(ray_indices) > 0:
        t_origins = batch_origins[ray_indices.long()]
        t_dirs = batch_directions[ray_indices.long()]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0

        predictions = get_predictions(coarse_model, positions, batch_size)

        coarse_pix_pred_vals, entropy_vals = acc_render_volume_density(predictions, ray_indices, t_starts, t_ends, img_sample_size, depth_samples_per_ray_coarse)
        # pdb.set_trace()
        pixel_loss_coarse = torch.nn.functional.mse_loss(coarse_pix_pred_vals, batch_pix_vals)
        # print(torch.mean(entropy_vals))

        # print(pixel_loss_coarse)
        loss_coarse = pixel_loss_coarse
        psnr_coarse = -10. * torch.log10(loss_coarse)

        coarse_optimizer.zero_grad()
        loss_coarse.backward()
        coarse_optimizer.step()

        # Log in tensorboard
        if n_iter % 100 == 0:
            writer.add_scalar('mean/train-pred-coarse', torch.mean(coarse_pix_pred_vals), n_iter)
            writer.add_scalar('mean/train', torch.mean(batch_pix_vals), n_iter)
            writer.add_scalar('loss/train-pixel-coarse', pixel_loss_coarse.item(), n_iter)
            writer.add_scalar('psnr/train-coarse', psnr_coarse.item(), n_iter)

            coarse_pred_resh = coarse_pix_pred_vals.reshape(sample_size, sample_size)
            batch_resh = batch_pix_vals.reshape(sample_size, sample_size)
            writer.add_image('Pred/train-pred-coarse', coarse_pred_resh, n_iter, dataformats='HW')
            writer.add_image('Orig/train', batch_resh, n_iter, dataformats='HW')
            writer.add_image('Diff/train-diff-coarse', torch.abs(coarse_pred_resh-batch_resh), n_iter, dataformats='HW')

        # update lr
        new_lr = lambda lr : lr * (decay_rate ** (n_iter / decay_steps))

        new_lr_coarse = new_lr(coarse_lr)

        for param_group in coarse_optimizer.param_groups:
            param_group['lr'] = new_lr_coarse

    if n_iter % display_every == 0:
        coarse_model.eval()
        print('lr coarse', new_lr_coarse)
        print('barf-coarse', coarse_model.barf_alpha) if coarse_model.use_pos_enc == 'barf' else None

        end_time = time.time()
        print(f'Time for iteration {n_iter} = {end_time - start_time}')

        with torch.no_grad():
            # nerf acc efficient raymarching
            ray_indices, t_starts, t_ends = acc_ray_marching(coarse_model, acc_grid, scene_aabb, test_origins, test_directions, depth_samples_per_ray_coarse, near_thresh, far_thresh, early_stop_eps, alpha_thre)

            t_origins = test_origins[ray_indices.long()]
            t_dirs = test_directions[ray_indices.long()]
            test_positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0

            if len(ray_indices) > 0:
                test_predictions = get_predictions(coarse_model, test_positions, batch_size)

                coarse_test_pix_pred_vals, entropy_vals = acc_render_volume_density(test_predictions, ray_indices, t_starts, t_ends, img_width*img_height, depth_samples_per_ray_coarse)

                # pdb.set_trace()
                coarse_test_pred_img[test_x_positions, test_y_positions] = coarse_test_pix_pred_vals.float()
                # print(np.mean(entropy_vals))

                pixel_loss_coarse = torch.nn.functional.mse_loss(coarse_test_pred_img, test_img)
                loss_coarse = pixel_loss_coarse
                psnr_coarse = -10. * torch.log10(loss_coarse)

                acc_grid_binary = acc_grid.binary.cpu().numpy().astype('int')
                pv_grid.cell_data['values'] = acc_grid_binary.flatten() 
                print(np.unique(acc_grid_binary.flatten(), return_counts=True))
                pv_grid.save(f"{log_dir}coarsegrid.vtk")

                vessel_acc_grid_binary = vessel_acc_grid.binary.cpu().numpy().astype('int')
                vessel_pv_grid.cell_data['values'] = vessel_acc_grid_binary.flatten() 
                print(np.unique(vessel_acc_grid_binary.flatten(), return_counts=True))
                vessel_pv_grid.save(f"{log_dir}coarsevesselgrid.vtk")

                
                test_pred_img_vessel = coarse_test_pred_img[test_dist_x_pos, test_dist_y_pos]
                vessel_psnr = None
                if len(test_pred_img_vessel) > 0:
                    vessel_pixel_loss = torch.nn.functional.mse_loss(test_pred_img_vessel, test_img_vessel)
                    vessel_psnr = -10. * torch.log10(vessel_pixel_loss)

                check_psnr = psnr_coarse if binary or sampling_strategy == 'random' else vessel_psnr

                if check_psnr >= highest_psnr and n_iter > 0:
                    highest_psnr = check_psnr
                    highest_iter = n_iter

                    coarse_model.save(f'{log_dir}highmodel.pth', {})
                    
                    pv_grid.save(f"{log_dir}highgrid.vtk")
                    vessel_pv_grid.save(f"{log_dir}highvesselgrid.vtk")

                    date_end = datetime.now().astimezone()

                    page_data['Date end'] = date_end.isoformat()
                    page_data['PSNR'] = np.around(float(psnr_coarse), decimals=2)
                    page_data['Vessel PSNR'] = np.around(float(vessel_psnr), decimals=2) if vessel_psnr else 0

                    with open(log_dir + 'readme.txt', 'w') as f:
                        for line in readme_lines:
                            f.write(line)
                            f.write('\n')
                        f.write(f'PSNR={psnr_coarse} end={date_end.strftime("%Y-%m-%d-%H%M")}')
                    
                    plt.imsave(log_dir + f'high-proj-{test_proj_id}.png', coarse_test_pred_img.cpu().numpy(), vmin=0, vmax=1, cmap='gray')
                    plt.imsave(log_dir + f'high-proj-{test_proj_id}-diff.png', torch.abs(coarse_test_pred_img-test_img).cpu().numpy(), cmap='gray', vmin=0, vmax=1)

                # log in tensorboard
                if n_iter % (display_every*2) == 0:
                    writer.add_scalar('loss/test-pixel-coarse', pixel_loss_coarse.item(), n_iter)
                    writer.add_scalar('psnr/test-coarse', psnr_coarse.item(), n_iter)
                    if vessel_psnr:
                        writer.add_scalar('psnr/vessel-test-coarse', vessel_psnr.item(), n_iter)

                    writer.add_scalar('barf-coarse', coarse_model.barf_alpha, n_iter) if coarse_model.use_pos_enc == 'barf' else None

                    writer.add_image('Pred/coarse-test-pred', coarse_test_pred_img, n_iter, dataformats='HW')
                    writer.add_image('Orig/test', test_img, dataformats='HW')
                    writer.add_image('Diff/coarse-test-diff', torch.abs(coarse_test_pred_img-test_img), n_iter, dataformats='HW')

                print("Iteration:", n_iter)
                print("Loss coarse:", loss_coarse.item())
                print("PSNR coarse:", psnr_coarse.item())
                if vessel_psnr:
                    print("Vessel coarse", vessel_psnr.item())
                
                if n_iter % save_every == 0:
                    plt.imsave(log_dir + f'coarse-proj-{test_proj_id}-{n_iter}.png', coarse_test_pred_img.cpu().numpy(), vmin=0, vmax=1, cmap='gray')
                    plt.imsave(log_dir + f'coarse-proj-{test_proj_id}-{n_iter}-diff.png', torch.abs(coarse_test_pred_img-test_img).cpu().numpy(), cmap='gray', vmin=0, vmax=1)

                    try:
                        coarse_model.save(f'{log_dir}coarsemodel.pth', {})
                        
                    except:
                        print('error saving model')
                
                print('Early stop iters:', n_iter - highest_iter)
                print(float(n_iter - highest_iter) >= float(early_stop_iters))
                # pdb.set_trace()
                if n_iter - highest_iter >= early_stop_iters:
                    print(f'Early stop = {n_iter}')
                    break

                if float(check_psnr) < float(highest_psnr) and n_iter - highest_iter >= early_stop_iters:
                    print(f'Early stop = {n_iter}')
                    break
    
    