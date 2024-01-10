import pandas as pd
import numpy as np
import torch

def map_cell_to_np(df, column_name, idx):
    column = pd.eval(df[column_name][idx])
    return np.array(column)

def get_query_points(x, y, img_width, img_height, focal_length, tform_cam2world, depth_samples_per_ray, near_thresh, far_thresh, device, randomize=False):
    direction = torch.stack([(x - img_width * .5) / focal_length,
                            -(y - img_height * .5) / focal_length,
                            -torch.ones_like(x)
                           ], dim=-1).to(device)
                           
    ray_directions = torch.sum(direction[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape).to(device)

    t_vals = torch.linspace(0., 1., depth_samples_per_ray)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)

    if randomize:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.concat([mids, z_vals[..., -1:]], -1)
        lower = torch.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(ray_origins.device)

    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    # query_points = query_points.cpu()
    return query_points, ray_origins, ray_directions, depth_values

def x_rotation_matrix(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0], 
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def y_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def z_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation_matrix(vec):
    m = np.identity(4)
    m[:3, 3] = vec[:3]
    return m

def get_rotation(theta, phi, larm, type='rotation'):
    # roadmap run geometry
    R = np.linalg.inv(z_rotation_matrix(np.deg2rad(larm)).dot(x_rotation_matrix(np.deg2rad(theta)).dot(y_rotation_matrix(np.deg2rad(phi)))))
    return R

def source_matrix(source_pt, theta, phi, larm=0, translation=[0,0,0], type='rotation'):
    m2 = get_rotation(theta, phi, larm)
    # translate back to source position
    m3 = translation_matrix(source_pt)
    # correct for table position
    m4 = translation_matrix([translation[0], translation[1], translation[2], 1])

    worldtocam = m4.dot(m2.dot(m3))

    return worldtocam