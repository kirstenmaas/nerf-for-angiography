import torch
import numpy as np
import imageio
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import griddata
import json
import ast
import os

def get_minibatches(inputs: torch.Tensor, chunksize= 1024 * 8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def get_predictions_vis(query_pts, model, chunksize, device):
    with torch.no_grad():
        pts_flatten = query_pts.reshape((-1, 3))

        batches = get_minibatches(pts_flatten, chunksize=chunksize)
        predictions = []
        for batch in batches:
            predictions.append(model(batch))
        raw = torch.cat(predictions, dim=0)

        if raw.shape[-1] == 2:
            sigma_a = torch.nn.functional.relu(raw[..., -1])
            opacs = sigma_a.detach()
            rgb = torch.sigmoid(raw[..., 0])
            rgb = rgb.detach()
        elif raw.shape[-1] > 2:
            sigma_a = torch.mean(torch.nn.functional.relu(raw), dim=-1)
            opacs = sigma_a.detach()
            rgb = opacs
        else:
            # sigma_a = torch.nn.functional.relu(raw[..., -1])
            sigma_a = torch.nn.Sigmoid()(raw[..., -1])
            opacs = sigma_a.detach()
            rgb = opacs
    return torch.stack([rgb, opacs]).to(device)

def save_video(frames, title, str, store_folder_name, fps=20, quality=7):
    f = f'{store_folder_name}{title}-{str}.mp4'
    imageio.mimwrite(f, frames, fps=10, quality=7, macro_block_size=10)

def get_videos(df, title, img_width, img_height, store_folder_name):
    gt_frames = []
    pred_frames = []
    diff_frames = []
    binary_pred_frames = []
    for index, row in df.iterrows():
        gt_img = np.array(row['org_img']).squeeze().reshape((img_width, img_height))
        pred_img = np.array(row['pred_img']).squeeze().reshape((img_width, img_height))
        binary_pred_img = np.array(row['binary_pred_img']).squeeze().reshape((img_width, img_height))
        diff_img = np.abs(gt_img - pred_img)
        
        gt_frames.append((255*np.clip(gt_img,0,1)).astype(np.uint8))
        pred_frames.append((255*np.clip(pred_img,0,1)).astype(np.uint8))
        diff_frames.append((255*np.clip(diff_img, 0, 1)).astype(np.uint8))
        binary_pred_frames.append((255*np.clip(binary_pred_img, 0, 1)).astype(np.uint8))
    
    save_video(gt_frames, title, 'gt', store_folder_name)
    save_video(pred_frames, title, 'pred', store_folder_name)
    save_video(diff_frames, title, 'diff', store_folder_name)
    save_video(binary_pred_frames, title, 'binary', store_folder_name)

def get_spherical_coordinates(thetas, phis):
    # Creating the coordinate grid for the unit sphere.
    # X = np.outer(np.sin(rad_thetas), np.cos(rad_phis))
    # Y = np.outer(np.sin(rad_thetas), np.sin(rad_phis))
    # Z = np.outer(np.cos(rad_thetas), np.ones(len(rad_phis)))

    coordinates = []
    angles = []
    for theta in thetas:
        for phi in phis:
            theta_rad = np.deg2rad(theta)
            phi_rad = np.deg2rad(phi)

            x = np.sin(theta_rad) * np.cos(phi_rad)
            y = np.sin(theta_rad) * np.sin(phi_rad)
            z = np.cos(theta_rad)
            coordinates.append([x,y,z])
            angles.append([theta, phi])
    coordinates = np.array(coordinates)
    angles = np.array(angles)

    return { 'X': coordinates[:,0], 'Y': coordinates[:,1], 'Z': coordinates[:,2], 'theta': angles[:,0], 'phi': angles[:,1] }

def convert_to_polar(x, y):
    theta = np.array(np.round(np.arctan2(y, x), decimals=2))
    r = np.array(np.round(np.sqrt(x**2 + y**2), decimals=2))
    return theta, r

def get_2d_heatmap(df, store_folder_name, store_info, name='top', x_axis='X', y_axis='Y', metric='PSNR', vminmax=[], center_point=[0,0], save_json=True):
    print(f'{name}-{x_axis}-{y_axis}-{metric}-{center_point}-save={save_json}')

    # TEST FOR X & Y TOP
    # df_copy = df_copy.sort_values(by=['theta', 'phi'])
    
    if (x_axis == 'X' and y_axis == 'Y') or (x_axis == 'Y' and y_axis == 'X'):
        if name == 'top':
            df = df[(df['theta'] <= 90) & (df['theta'] >= -90) & (df['phi'] <= 90) & (df['phi'] >= -90)]
        elif name == 'bottom':
            df = df[((df['theta'] >= 90) | (df['theta'] <= -90)) & ((df['phi'] >= 90) | (df['phi'] <= -90))]
    elif (x_axis == 'X' and y_axis == 'Z') or (x_axis == 'Z' and y_axis == 'X'):
        if name == 'top':
            df = df[(df['theta'] >= 0) & (df['theta'] <= 180) & (df['phi'] <= 90) & (df['phi'] >= -90)]
        elif name == 'bottom':
            df = df[((df['theta'] <= 0) & (df['theta'] >= -180)) & (df['phi'] <= 90) & (df['phi'] >= -90)]
    elif (x_axis == 'Y' and y_axis == 'Z') or (x_axis == 'Z' and y_axis == 'Y'):
        if name == 'top':
            df = df[(df['theta'] <= 90) & (df['theta'] >= -90) & (df['phi'] >= 0) & (df['phi'] <= 180)]
        elif name == 'bottom':
            df = df[(df['theta'] <= 90) & (df['theta'] >= -90) & ((df['phi'] <= 0) & (df['phi'] >= -180))]

    interest_thetas = pd.unique(df['theta'])
    interest_phis = pd.unique(df['phi'])

    theta, rad = convert_to_polar(df[f'cam_pose_{x_axis.lower()}'], df[f'cam_pose_{y_axis.lower()}']+1e-10)

    # # theta 
    # rads_0_idx = np.argwhere(rad == 0).flatten()
    # theta[rads_0_idx] = theta[:len(rads_0_idx)]

    df.loc[:, 'theta_polar'] = theta
    df.loc[:, 'rad_polar'] = rad

    theta_r = theta.reshape(len(interest_phis), len(interest_thetas))
    rad_r = rad.reshape(theta_r.shape)

    vals_r = []
    angles_r = []
    all_vals_r = []
    imgs_flatten_r = []
    angles_flatten_r = []

    for ii in range(theta_r.shape[0]):
        vals_r_i = []
        angles = []
        for jj in range(theta_r.shape[1]):
            row = df[(df['theta_polar'] == theta_r[ii, jj]) & (df['rad_polar'] == rad_r[ii,jj])]#.iloc[0]

            if len(row) == theta_r.shape[1]:
                row = row.iloc[jj]
            else:
                row = row.iloc[0]
            
            val = np.min(row[metric].tolist()) # in case of multiple values for polar coordinates
            angles.append([float(row['theta']), float(row['phi'])])


            if jj < theta_r.shape[1] - 1 and not np.array_equal(np.unique(rad_r[ii]), [0.]):
                vals_r_i.append(val)
            all_vals_r.append(val)

            # print(row['pred_img'].dtype)
            
            pred_img = row['pred_img']
            org_img = row['org_img']
            if type(row['pred_img']) == str:
                pred_img = ast.literal_eval(pred_img)
                org_img = ast.literal_eval(org_img)
            diff_img = np.abs(np.array(pred_img) - np.array(org_img))

            imgs_flatten_r.append([pred_img, org_img, diff_img])
            angles_flatten_r.append([float(row['theta']), float(row['phi'])])

        if len(vals_r_i) > 0:
            vals_r.append(np.array(vals_r_i))
        angles_r.append(angles)
        
    all_vals_r = np.array(all_vals_r).reshape(theta_r.shape)
    vals_r = np.array(vals_r)
    angles_r = np.array(angles_r)
    imgs_flatten_r = np.array(imgs_flatten_r)
    angles_flatten_r = np.array(angles_flatten_r)

    vminmaxstr = '-'
    if len(vminmax) == 2:
        vminmaxstr = f'-{np.round(vminmax, decimals=2)}'
    
    plt.figure(figsize=(30, 30))
    plt.subplot(projection="polar")

    if vals_r.shape[0] == theta_r.shape[0] - 1 and vals_r.shape[1] == theta_r.shape[1] - 1:
        # plot centerpoint location
        row_c = df[(df['theta'] == center_point[0]) & (df['phi'] == center_point[1])]
        
        # plt.scatter(float(row_c[f'cam_pose_{x_axis.lower()}']), float(row_c[f'cam_pose_{y_axis.lower()}']), c='black', s=100)
        # plt.scatter(float(row_c[f'theta_polar']), float(row_c[f'rad_polar'])+0.1, c='black', s=100)

        plt.pcolormesh(theta_r, rad_r, vals_r, vmin=vminmax[0], vmax=vminmax[1], alpha=0.9)
        
        # plt.colorbar()

        if len(row_c) > 0:
            row_c = row_c.iloc[0]
            plt.scatter(float(row_c[f'theta_polar']), float(row_c[f'rad_polar']), c='black', s=100)

        rad_ticks = np.unique(rad_r)
        angle_ticks = np.unique(np.abs(angles_r[:,0]))
        if angle_ticks[0] == 0:
            rad_ticks = np.insert(rad_ticks, 1, 0.04)
            angle_ticks = np.insert(angle_ticks, 1, 5)
        plt.yticks(rad_ticks, angle_ticks)

        for ii in range(rad_r.shape[0]):
            for jj in range(rad_r.shape[1]):
                # if np.any(np.all(angles_r[ii, jj] == gt_angles, axis=1)):
                plt.text(theta_r[ii, jj], rad_r[ii, jj], angles_r[ii, jj], color='red')

        # label each cell?
        # for ii,i in enumerate(df.index):
        #     for jj,j in enumerate(df.keys()):
        #         plt.text(ii+0.5,jj+0.5,df[i][j])
        
        plt.savefig(f'{store_folder_name}/heatmap-{metric}-{name}{vminmaxstr}-{x_axis}-{y_axis}.png')
        plt.close()
    else:
        print('axes did not match')
    
    if save_json:
        # save vals for javascript heatmap
        # sort such that rad == 0 is last
        json_rad = rad_r.flatten()
        rad_idx = np.argsort(json_rad)[::-1]

        json_rad = json_rad[rad_idx]
        json_theta = theta_r.flatten()[rad_idx]
        json_angles = angles_flatten_r[rad_idx]
        json_vals = all_vals_r.flatten()[rad_idx]
        json_pred_imgs = imgs_flatten_r[rad_idx]

        experiment_folder = f"{store_info['json_file_path']}/{store_info['experiment']}/{store_info['experiment_name']}"

        if not os.path.exists(experiment_folder):
            os.mkdir(experiment_folder)
        
        json_object = { 'rad': json_rad.tolist(), 'theta': json_theta.tolist(), 'angles': json_angles.tolist(), 'vals': json_vals.tolist() }#, 'predImg': json_pred_imgs.tolist() }

        json_file_name = f"{experiment_folder}/{metric}-{name}-{x_axis}-{y_axis}.json"
        
        with open(json_file_name, 'w') as f:
            json.dump(json_object, f)

        # save per unique theta angle
        # imgs_flatten_r = imgs_flatten_r.reshape(angles_r.shape[0], angles_r.shape[1], imgs_flatten_r.shape[-2], imgs_flatten_r.shape[-1])
        # pdb.set_trace()
        for kdx, angles in enumerate(json_angles):
            file_name = f'{angles[0]}{angles[1]}.json'
            angle_json_object = { 'pred': json_pred_imgs[kdx, 0].tolist(), 'org': json_pred_imgs[kdx, 1].tolist(), 'diff': json_pred_imgs[kdx, 2].tolist()}#, 'phi': phi_angles.tolist() }
            with open(f"{experiment_folder}/{file_name}", 'w') as f:
                json.dump(angle_json_object, f)
        # pdb.set_trace()
        # plt.imsave(f'{experiment_folder}/{angles_flatten_r[0][0]}{angles_flatten_r[0][1]}.png', json_pred_imgs[0, 0].reshape(int(np.sqrt(json_pred_imgs[0, 0].shape)), -1))