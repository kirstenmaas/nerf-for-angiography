import numpy as np

import pyvista as pv
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from skimage import io
from torchmetrics import StructuralSimilarityIndexMeasure, Dice
from ast import literal_eval
import copy
import matplotlib as mpl
import pdb
import argparse
from tqdm import tqdm
from nerfacc import OccupancyGrid, ContractionType
import gc

from piq import LPIPS, DISTS

from model.CPPN import CPPN

from phantomdata.helpers import get_ray_values, get_depth_values, get_interpolator_from_grid
from visualization.helpers import get_videos, get_2d_heatmap, get_predictions_vis

from nerf.nerf_helpers import get_predictions
from nerf.nerf_helpers_acc import acc_ray_marching, acc_render_volume_density

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

chunksize = 16384 #1024

# what to compute
save_vtk = True
# metrics = ['DISTS', 'LPIPS']
metrics = ['DISTS', 'LPIPS', 'PSNR'] #'PSNR', 'SSIM', 'DICE 2D', 'DOT 2D', 'DICE 3D', 'DOT 3D']
heatmap_metrics = ['LPIPS', 'DISTS']
vis_limited_angles = [5, 15, 30, 45, 60, 90, 180]

save_videos = True
save_heatmap = True

compute_metrics = True if len(metrics) > 0 else False

parser = argparse.ArgumentParser()

parser.add_argument('--binary', help='Whether images are binary or not')
parser.add_argument('--data_name', help='Either CT data or LCA data')

args = parser.parse_args()
arg_dict = vars(args)

# define general parameters
binary = arg_dict['binary'] == 'True' or arg_dict['binary'] == True if arg_dict['binary'] is not None else False
data_name = arg_dict['data_name'] if arg_dict['data_name'] else 'ct'

folder_name = 'runs'
model_type = 'coarse' #'coarse'
translation = np.array([0, 0, 0])

limited_size_vis = 360
number_angles_vis = 36
angle_step_vis = limited_size_vis/number_angles_vis

depth_samples_per_ray = 400
outside = 100

scene_aabb = torch.tensor([-outside, -outside, -outside, outside, outside, outside], dtype=torch.float32, device=device)
acc_grid = OccupancyGrid(roi_aabb=scene_aabb, resolution=128, contraction_type=ContractionType.AABB).to(device)

cathlab = False

if data_name == 'ct':
    focal_length = 1300
    src_pt = np.array([0, 0, focal_length+200])
    img_width = 100
    img_height = 100

    sample_outside = 75
    depth_samples_per_ray = 200

    data_file_name = data_name
    
elif data_name == 'LCA':
    focal_length = 4000
    src_pt = np.array([0, 0, focal_length])

    img_width = 150
    img_height = 162

    sample_outside = 80
    outside = 80
    
    depth_samples_per_ray = 200

    data_file_name = f'stl/{data_name}'

near_vol_thresh = -outside
far_vol_thresh = outside
t = np.linspace(near_vol_thresh, far_vol_thresh, depth_samples_per_ray+1)

near_thresh = src_pt[2] - sample_outside
far_thresh = src_pt[2] + sample_outside

data_folder_name = os.path.join(os.getcwd(), f'data/{data_file_name}/projections')
binary_data_folder_name = os.path.join(os.getcwd(), f'data/{data_file_name}/projections')
if not binary:
    data_folder_name = os.path.join(os.getcwd(), f'data/{data_file_name}/projections-background')

if cathlab:
    data_folder_name = os.path.join(os.getcwd(), f'data/{data_file_name}/cttoproj_standardviews_background')
    if binary:
        data_folder_name = os.path.join(os.getcwd(), f'data/{data_file_name}/cttoproj_standardviews')

time_strs = os.listdir(f'cases/{data_name}/{folder_name}')[::-1]

for time in time_strs:
    print(time)
    store_folder_name = os.path.join(os.getcwd(), f'cases/{data_name}/{folder_name}/{time}/')

    # this code should load the properties saved during training
    page_data = None
    if not page_data:
        print(time, 'not found')
        continue
    else:
        page_data = page_data['properties']

    # check if binary setting matches and the data matches
    if binary != bool(page_data['Binary']['checkbox']) or data_name.upper() != page_data['Data']['select']['name']:
        print(time, 'does not match binary setting or data type')
        continue

    categories = [category['name'] for category in page_data['Category']['multi_select']]
    sampling_strategies = [sampling_strategy['name'] for sampling_strategy in page_data['Sampling']['multi_select']]

    model_architecture = page_data['Model architecture']['select']['name']

    # pdb.set_trace()
    if int(page_data['Limited projections']['select']['name']) not in vis_limited_angles or model_architecture != '4x128': #or (not (center_point[0] == 90 and center_point[1] == 0)):
        print('limited projections too low of interest')
        print(center_point)
        continue
    org_df = []

    # check if proj folder exists
    proj_folder_name = os.path.join(os.getcwd(), f'{store_folder_name}/projections/')
    if not os.path.isdir(proj_folder_name):
        os.mkdir(proj_folder_name)
    # check if proj folder exists
    org_folder_name = os.path.join(os.getcwd(), f'{store_folder_name}/originals/')
    if not os.path.isdir(org_folder_name):
        os.mkdir(org_folder_name)

    # print('reading grid...')
    grid_reader = pv.get_reader(f'{store_folder_name}/{model_type}grid.vtk')
    grid_grid = grid_reader.read()
    grid_occupancy = torch.from_numpy(grid_grid['values'].reshape(np.array(grid_grid.dimensions) - 1)).to(device).bool()
    
    acc_grid._binary = grid_occupancy

    # load GT 3D vol
    print('reading 3d volume...')
    vol_reader = pv.get_reader(data_folder_name + '/ground-truth.vtk')
    
    vol_grid = vol_reader.read()
    gt_interpolator, _, _ = get_interpolator_from_grid(vol_grid, binary=binary)

    binary_scalars = vol_grid.point_data['scalars']
    binary_thresh = 0.05
    binary_grid = vol_reader.read()

    binary_scalars[binary_scalars < binary_thresh] = 0
    binary_grid.point_data['scalars'] = binary_scalars
    gt_binary_interpolator, _, _ = get_interpolator_from_grid(binary_grid, binary=binary)

    model_info = torch.load(os.path.join(store_folder_name, f'{model_type}model.pth'), map_location=device)
    model_info['parameters']['act_func'] = 'relu'
    model = CPPN(model_info['parameters'])

    model.to(device)

    model.load_state_dict(model_info['model']) #map_location=torch.device('cpu')))
    model.eval()

    th_angles = np.arange(-limited_size_vis//2, limited_size_vis//2+1, angle_step_vis).astype('float64')
    ph_angles = np.arange(-limited_size_vis//2, limited_size_vis//2+1, angle_step_vis).astype('float64')

    th_ph_angles = np.array([np.array(val) for val in itertools.product(th_angles, ph_angles)])

    # check if all angles already appear in org df so we don't have to loop over everything
    skip_preds = False
    if len(org_df) > 0:
        df_angles = org_df[['theta', 'phi']].to_numpy()
        
        # check if intersection has same shape as th_ph_angles
        inters_angles = np.array([x for x in set(tuple(x) for x in th_ph_angles) & set(tuple(x) for x in df_angles)])
        if th_ph_angles.shape == inters_angles.shape:
            skip_preds = True

    # obtain 3D point cloud
    opacs_pred = None
    opacs_gt = None
    if (not skip_preds):
        print('generating pointcloud...')

        mesh_grid = np.meshgrid(t, t, t)
        query_pts = torch.Tensor(np.stack(mesh_grid, -1)).to(device)
        flat_query_pts = query_pts.reshape((-1, 3))

        with torch.no_grad():
            occ_pts = acc_grid.query_occ(flat_query_pts)
            opacs_pred = torch.zeros(occ_pts.shape).to(device)

            if data_name == 'LCA':
                # check_pts_idx = torch.where(occ_pts)
                check_pts_idx = torch.arange(0, occ_pts.shape[0])
                check_pts = flat_query_pts[check_pts_idx]
                # print(f'{check_pts.shape}/{occ_pts.shape}')
                # pdb.set_trace()
            else:
                check_pts_idx = torch.where(occ_pts >= 0)
                check_pts = flat_query_pts[check_pts_idx]
                print(f'{check_pts.shape}/{occ_pts.shape}')

            _, opacs_check_pred = get_predictions_vis(check_pts, model, chunksize, device)
            opacs_pred[check_pts_idx] = opacs_check_pred

            opacs_gt = torch.Tensor(gt_interpolator(query_pts.reshape((-1, 3)).cpu().numpy())).to(device)
        # pdb.set_trace()

        if save_vtk:
            grid = pv.StructuredGrid(mesh_grid[0], mesh_grid[1], mesh_grid[2])
            grid.point_data['scalars'] = opacs_pred.cpu().numpy()
            grid.save(f"{store_folder_name}{model_type}{time}.vtk")
        print('obtained pointcloud!')
        pdb.set_trace()

    dice = Dice(average='micro').to(device) #calculate dice globally, independent of classes

    psnrs = {}
    ssims = {}
    dice_2ds = {}
    dots_2ds = {}
    lpipss = {}
    distss = {}
    

    preds_imgs = {}
    org_imgs = {}
    binary_preds_imgs = {}

    thetas = {}
    phis = {}
    larms = {}

    thetas_360 = {}
    phis_360 = {}

    cam_poses_x = {}
    cam_poses_y = {}
    cam_poses_z = {}

    if 'SSIM' in metrics:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if 'LPIPS' in metrics:
        lpips = LPIPS().to(device)

    if 'DISTS' in metrics:
        dists = DISTS().to(device)

    if not skip_preds:
        print('generating images...')
        for angles in tqdm(th_ph_angles):
            theta, phi = angles

            theta_360 = theta if theta >= 0 else 360 + theta
            phi_360 = phi if phi >= 0 else 360 + phi

            image_id = f'{theta}-{phi}'.replace('.', ',')
            file_image_id = f'image-{theta}-{phi}-{0}'
            target_img = torch.Tensor(io.imread(f'{data_folder_name}/{file_image_id}.png', as_gray=True)).to(device).float()
            np_target_img = copy.deepcopy(target_img.cpu().numpy())
            binary_target_img = torch.Tensor(io.imread(f'{binary_data_folder_name}/{file_image_id}.png', as_gray=True)).to(device).float()

            thetas[image_id] = theta
            phis[image_id] = phi
            larms[image_id] = 0

            thetas_360[image_id] = theta_360
            phis_360[image_id] = phi_360

            org_imgs[image_id] = np_target_img.reshape(-1).tolist()
            # plt.imsave(f'{org_folder_name}{file_image_id}.png', target_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)

            # check if image_id already in org df
            org_row = org_df[org_df['image_id'] == image_id] if len(org_df) > 0 else []
            if len(org_row) > 0 and os.path.isfile(f'{proj_folder_name}{file_image_id}-binary.png'):
                org_row = org_row.iloc[0]
                
                # get cam poses from org df
                cam_poses_x[image_id] = org_row['cam_pose_x']
                cam_poses_y[image_id] = org_row['cam_pose_y']
                cam_poses_z[image_id] = org_row['cam_pose_z']

                # load preds img
                np_preds_img = np.array(io.imread(f'{proj_folder_name}{file_image_id}.png', as_gray=True))
                np_binary_preds_img = np.array(io.imread(f'{proj_folder_name}{file_image_id}-binary.png', as_gray=True))
                preds_img = torch.Tensor(np_preds_img).to(device).float()
                binary_preds_img = torch.Tensor(np_binary_preds_img).to(device).float()
            else:
                ray_origins, ray_directions, src_matrix, _, _ = get_ray_values(theta_360, phi_360, 0, src_pt, img_width, img_height, focal_length, device, translation)
                depth_values = get_depth_values(near_thresh, far_thresh, depth_samples_per_ray, device, stratified=False)

                cam_pose = src_matrix[:3, -1]
                cam_poses_x[image_id] = cam_pose[0]
                cam_poses_y[image_id] = cam_pose[1]
                cam_poses_z[image_id] = cam_pose[2]

                if os.path.isfile(f'{proj_folder_name}{file_image_id}.png') and os.path.isfile(f'{proj_folder_name}{file_image_id}-binary.png'):
                    np_preds_img = np.array(io.imread(f'{proj_folder_name}{file_image_id}.png', as_gray=True))
                    np_binary_preds_img = np.array(io.imread(f'{proj_folder_name}{file_image_id}-binary.png', as_gray=True))
                    preds_img = torch.Tensor(np_preds_img).to(device).float()
                    binary_preds_img = torch.Tensor(np_binary_preds_img).to(device).float()
                else:
                    if data_name == 'ct':
                        with torch.no_grad():
                            ray_origins = ray_origins.reshape((-1, 3)).float()
                            ray_directions = ray_directions.reshape((-1, 3)).float()
                            scene_aabb = scene_aabb.float()

                            ray_indices, t_starts, t_ends = acc_ray_marching(model, acc_grid, scene_aabb, ray_origins, ray_directions, depth_samples_per_ray, near_thresh, far_thresh)
                            t_origins = ray_origins[ray_indices.long()]
                            t_dirs = ray_directions[ray_indices.long()]
                            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
                            # pdb.set_trace()

                            predictions = get_predictions(model, positions, chunksize)

                            coarse_pix_pred_vals, _ = acc_render_volume_density(predictions, ray_indices, t_starts, t_ends, img_width*img_height, depth_samples_per_ray)

                            preds_img = coarse_pix_pred_vals.reshape(target_img.shape)
                            np_preds_img = copy.deepcopy(preds_img.cpu().numpy())

                            binary_predictions = torch.Tensor(predictions)
                            sigma_binary_predictions = torch.nn.Sigmoid()(binary_predictions)
                            sigma_idx = torch.where(sigma_binary_predictions < binary_thresh)
                            
                            binary_pix_pred_vals, _ = acc_render_volume_density(binary_predictions, ray_indices, t_starts, t_ends, img_width*img_height, depth_samples_per_ray, sigma_idx)

                            binary_preds_img = binary_pix_pred_vals.reshape(target_img.shape)
                            np_binary_preds_img = copy.deepcopy(binary_preds_img.cpu().numpy())
                    else:
                        with torch.no_grad():
                            query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
                            ray_points = query_points.reshape((-1, 3)).to(device).float()

                            pred_vals = torch.zeros(ray_points.shape[0]).to(device)
                            binary_pred_vals = torch.zeros(ray_points.shape[0]).to(device)
                            # occ_vals = acc_grid.query_occ(ray_points)

                            check_pts_idx = torch.arange(0, pred_vals.shape[0])
                            check_pts = ray_points[check_pts_idx]

                            # check_pts_idx = torch.where(occ_vals)
                            # check_pts = ray_points[check_pts_idx]
                            # pdb.set_trace()

                            # model, flattened_query_points, chunksize, target_img_idx=None
                            pred_points, _ = get_predictions_vis(check_pts, model, chunksize, device)#, device)
                            pred_vals[check_pts_idx] = pred_points.flatten()

                            binary_predictions = torch.Tensor(pred_points)
                            binary_predictions[binary_predictions < binary_thresh] = 0

                            binary_pred_vals[check_pts_idx] = binary_predictions.flatten()

                            
                            flat_pred_points = pred_vals.reshape((ray_origins.shape[0], ray_origins.shape[1], depth_samples_per_ray))

                            flat_binary_pred_points = binary_pred_vals.reshape(flat_pred_points.shape)
                            
                            one_e_10 = torch.tensor([1e10], dtype=ray_directions.dtype, device=ray_directions.device)
                            dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

                            dists = dists.repeat(ray_origins.shape[0], ray_origins.shape[1], 1)

                            alphas = torch.exp(-flat_pred_points * dists)
                            preds_img = torch.prod(alphas, dim=-1).float()
                            np_preds_img = copy.deepcopy(preds_img.cpu().numpy())

                            binary_alphas = torch.exp(-flat_binary_pred_points * dists)
                            binary_preds_img = torch.prod(binary_alphas, dim=-1).float()
                            np_binary_preds_img = copy.deepcopy(binary_preds_img.cpu().numpy())

                    plt.imsave(f'{proj_folder_name}{file_image_id}.png', preds_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    plt.imsave(f'{proj_folder_name}{file_image_id}-binary.png', binary_preds_img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    plt.close()

            preds_imgs[image_id] = np.round(np_preds_img.reshape(-1), decimals=10).tolist()
            binary_preds_imgs[image_id] = np.round(np_binary_preds_img.reshape(-1), decimals=10).tolist()

            if 'PSNR' in metrics:
                pixel_loss = torch.nn.functional.mse_loss(preds_img, target_img)
                psnr_value = -10. * torch.log10(pixel_loss)
                psnrs[image_id] = float(psnr_value.cpu().numpy())

            if 'SSIM' in metrics:
                # ssim requires shape (BxCxHxW)
                ssim_preds_img = preds_img.reshape((1, 1, preds_img.shape[0], preds_img.shape[1]))
                ssim_target_img = target_img.reshape(ssim_preds_img.shape)

                ssim_value = ssim(ssim_preds_img, ssim_target_img)
                ssims[image_id] = float(ssim_value.cpu().numpy())

            if 'LPIPS' in metrics:
                lpips_preds_img = preds_img.reshape((1, 1, preds_img.shape[0], preds_img.shape[1]))
                lpips_target_img = target_img.reshape(lpips_preds_img.shape)

                lpips_loss = lpips(lpips_preds_img, lpips_target_img)

                lpipss[image_id] = float(lpips_loss.cpu().numpy())

            if 'DISTS' in metrics:
                dists_preds_img = preds_img.reshape((1, 1, preds_img.shape[0], preds_img.shape[1]))
                dists_target_img = target_img.reshape(lpips_preds_img.shape)

                dists_loss = dists(dists_preds_img, dists_target_img)

                distss[image_id] = float(dists_loss.cpu().numpy())

            if 'DICE 2D' in metrics:
                binary_preds_img[binary_preds_img < 1] = 0
                binary_target_img[binary_target_img < 1] = 0

                dice_2d_score = float(dice(binary_preds_img.int(), binary_target_img.int()))
                dice_2ds[image_id] = dice_2d_score
            
            if 'DOT 2D' in metrics:
                dot_2d_score = float(torch.mean(preds_img * target_img))
                # dots_2ds[image_id] = dot_2d_score

                # vs. normalization dot 2d
                norm_preds_img = preds_img - torch.min(preds_img)
                norm_preds_img /= torch.max(norm_preds_img)

                norm_target_img = target_img - torch.min(target_img)
                norm_target_img /= torch.max(norm_target_img)

                norm_dot_2d_score = float(torch.mean(norm_preds_img * norm_target_img))
                dots_2ds[image_id] = norm_dot_2d_score

        print('creating df')
        df = pd.DataFrame({'image_id': lpipss.keys(), 'theta': thetas.values(), 'phi': phis.values(), 'larm': larms.values(), 
            'theta_360': thetas_360.values(), 'phi_360': phis_360.values(),
            'cam_pose_x': cam_poses_x.values(), 'cam_pose_y': cam_poses_y.values(), 'cam_pose_z': cam_poses_z.values() })
        print('created df')

        if 'PSNR' in metrics:
            df['PSNR'] = psnrs.values()

        if 'SSIM' in metrics:
            df['SSIM'] = ssims.values()

        if 'LPIPS' in metrics:
            df['LPIPS'] = lpipss.values()

        if 'DISTS' in metrics:
            df['DISTS'] = distss.values()

        if 'DICE 2D' in metrics:
            df['DICE 2D'] = dice_2ds.values()

        if 'DOT 2D' in metrics:
            df['DOT 2D'] = dots_2ds.values()

        if 'DICE 3D' in metrics:
            # use the MEAN of the target volume to compute dice

            bin_opacs_pred = torch.ones(opacs_pred.shape)
            bin_opacs_pred[opacs_pred < torch.mean(opacs_gt)] = 0

            bin_opacs_gt = torch.ones(opacs_gt.shape)
            bin_opacs_gt[opacs_gt < torch.mean(opacs_gt)] = 0

            dice_3d_score = float(dice(bin_opacs_pred.int(), bin_opacs_gt.int()))
            df['DICE 3D'] = np.repeat(dice_3d_score, len(psnrs.values()))

        # TODO: check whether this makes sense!
        if 'DOT 3D' in metrics:
            dot_3d_score = float(torch.mean(opacs_pred * opacs_gt))
            df['DOT 3D'] = np.repeat(dot_3d_score, len(psnrs.values()))

            # vs. normalization dot 2d
            norm_opacs_pred = opacs_pred - torch.min(opacs_pred)
            norm_opacs_pred /= torch.max(norm_opacs_pred)

            norm_opacs_gt = opacs_gt - torch.min(opacs_gt)
            norm_opacs_gt /= torch.max(norm_opacs_gt)

            norm_dot_2d_score = float(torch.mean(norm_preds_img * norm_target_img))
            dots_2ds[image_id] = norm_dot_2d_score

        df.to_csv(f'{store_folder_name}/df-metrics.csv', sep=';')
        print('saved df')

        # store images separately so they don't get saved in the csv
        df['pred_img'] = preds_imgs.values()
        df['org_img'] = org_imgs.values()
        df['binary_pred_img'] = binary_preds_imgs.values()

        df['pred_img'] = df['pred_img'].map(list)
        df['org_img'] = df['org_img'].map(list)
        

        metrics_save = ['PSNR', 'DISTS', 'LPIPS']#, 'PSNR', 'SSIM', 'DOT 2D', 'DICE 2D']
        metric_types = ['min', 'mean', 'std']
        metric_page_data = {}

        # 2D metrics
        for metrics_save in metrics_save:
            for metric_type in metric_types:
                metric_value = None
                if metric_type == 'min':
                    value = np.min(df[metrics_save])
                elif metric_type == 'mean':
                    value = np.mean(df[metrics_save])
                elif metric_type == 'std':
                    value = np.std(df[metrics_save])
                value = round(float(value), 6)

                metric_page_data[f'{metrics_save} {metric_type}'] = value

        if save_videos:
            # theta rotation
            title = 'theta-rotation'
            theta_rows = df.loc[df['phi'] == 0.0]
            get_videos(theta_rows, title, img_height, img_width, proj_folder_name)

            # phi rotation
            title='phi-rotation'
            phi_rows = df.loc[df['theta'] == 0.0]
            get_videos(phi_rows, title, img_height, img_width, proj_folder_name)

    else:
        print('skip pred rendering')
        df = org_df.copy()

        # we still need to load the pred and org imgs from the files
        preds_imgs = {}
        binary_preds_imgs = {}
        for angles in tqdm(th_ph_angles):
            theta, phi = angles
            image_id = f'{theta}-{phi}'.replace('.', ',')
            file_image_id = f'image-{theta}-{phi}-{0}'
            
            np_preds_img = np.array(io.imread(f'{proj_folder_name}{file_image_id}.png', as_gray=True))
            np_binary_preds_img = np.array(io.imread(f'{proj_folder_name}{file_image_id}-binary.png', as_gray=True))

            preds_imgs[image_id] = np.round(np_preds_img.reshape(-1), decimals=10).tolist()
            binary_preds_imgs[image_id] = np.round(np_binary_preds_img.reshape(-1), decimals=10).tolist()

            np_target_img = io.imread(f'{data_folder_name}/{file_image_id}.png', as_gray=True)
            org_imgs[image_id] = np_target_img.reshape(-1).tolist()
        df['pred_img'] = preds_imgs.values()
        df['binary_pred_img'] = binary_preds_imgs.values()
        df['org_img'] = org_imgs.values()

    if save_heatmap:
        metric_arrays = { 'PSNR': psnrs, 'LPIPS': lpipss, 'DISTS': distss }

        # create min-max dictionary
        min_max_dict = {}
        for heatmap_metric in heatmap_metrics:
            min_max_dict[f'{heatmap_metric}-min'] = np.min(metric_arrays[heatmap_metric])
            min_max_dict[f'{heatmap_metric}-max'] = np.max(metric_arrays[heatmap_metric])

        df['cam_pose_x'] = ((df['cam_pose_x'] - np.min(df['cam_pose_x'])) / (np.max(df['cam_pose_x']) - np.min(df['cam_pose_x']))) * 2 - 1
        df['cam_pose_y'] = ((df['cam_pose_y'] - np.min(df['cam_pose_y'])) / (np.max(df['cam_pose_y']) - np.min(df['cam_pose_y']))) * 2 - 1
        df['cam_pose_z'] = ((df['cam_pose_z'] - np.min(df['cam_pose_z'])) / (np.max(df['cam_pose_z']) - np.min(df['cam_pose_z']))) * 2 - 1

        # note down the ground truth training angles
        try:
            gt_nmb_angles = int(np.sqrt(int(page_data['Sparse projections']['select']['name']))-1)
            gt_limited_size = int(page_data['Limited projections']['select']['name'])
        except Exception as e:
            print(e, page_data)
            gt_nmb_angles = 1
            gt_limited_size = 1

        # store based on experiment parameters
        experiment = ''
        experiment_name = ''
        if 'Limited projections' in categories and 'Sparse projections' in categories:
            experiment = 'limited-sparse'
            experiment_name = f'{gt_limited_size}-{gt_nmb_angles}-{center_point}'
        elif 'Background' in categories and len(categories) == 1:
            experiment = 'background'
            experiment_name = f'{gt_limited_size}-{gt_nmb_angles}-{center_point}'
            if 'Random sampling' in sampling_strategies:
                experiment += '-random'
            elif 'Segmentation sampling' in sampling_strategies:
                experiment += '-segmentation'

        elif 'Sparsity' in categories and len(categories) == 1:
            experiment = 'sparsity'
            experiment_name = f'{gt_limited_size}-{gt_nmb_angles}-{center_point}'
            

            if 'Random sampling' in sampling_strategies:
                experiment += '-random'
            elif 'Segmentation sampling' in sampling_strategies:
                experiment += '-segmentation'
        else:
            experiment = f'architecture-{model_architecture}'
            experiment_name = f'{gt_limited_size}-{gt_nmb_angles}-{center_point}'

        if 'LCA' in page_data['Data']['select']['name']:
            experiment += '-lca'
        elif 'CT' in page_data['Data']['select']['name']:
            experiment += '-ct'

        store_info = {
            'experiment': experiment,
            'experiment_name': experiment_name,
            'json_file_path': '/jsonData/',
        }

        first_axes = ['X', 'Y']
        second_axes = ['X', 'Z']
        third_axes = ['Y', 'Z']

        experiment_folder = f"{store_info['json_file_path']}/{store_info['experiment']}/{store_info['experiment_name']}"
        # create folder if it does not exist
        if not os.path.isdir(f"{store_info['json_file_path']}/{store_info['experiment']}"):
            os.mkdir(f"{store_info['json_file_path']}/{store_info['experiment']}")

        for axes in [second_axes]:
            for metric_name in heatmap_metrics:
                vmin, vmax = 0, 1
                if metric_name == 'PSNR':
                    vmin, vmax = 15, 50
                elif metric_name == 'SSIM':
                    vmin, vmax = 0.8, 1
                elif metric_name == 'DICE 2D':
                    vmin, vmax = 0.3, 1
                
                json_file_name = f"{experiment_folder}/{metric_name}-{'top'}-{axes[0]}-{axes[1]}.json"
                if not os.path.isfile(json_file_name):
                    get_2d_heatmap(df, store_folder_name, store_info, name='top', metric=metric_name, x_axis=axes[0], y_axis=axes[1], vminmax=[vmin, vmax], center_point=center_point)

                json_file_name = f"{experiment_folder}/{metric_name}-{'bottom'}-{axes[0]}-{axes[1]}.json"
                if not os.path.isfile(json_file_name):
                    get_2d_heatmap(df, store_folder_name, store_info, name='bottom', metric=metric_name, x_axis=axes[0], y_axis=axes[1], vminmax=[vmin, vmax], center_point=center_point)    

    del df
    if opacs_pred is not None:
        del opacs_pred
        del opacs_gt

    plt.close()

    gc.collect()