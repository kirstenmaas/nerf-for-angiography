import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import pdb

from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import frangi
from scipy.ndimage import distance_transform_edt

try:
    from .proj_helpers import source_matrix
except:
    from proj_helpers import source_matrix

def rev_sigmoid(x, c1=1, c2=0):
    return 1/(1+np.exp(c1*(x-c2)))

def line(vals, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    sub_vals = vals[(vals >= x1) & (vals < x2)]
    
    m = (y1-y2)/(x1-x2)
    b = (x1*y2 - x2*y1)/(x1-x2)

    return m*sub_vals + b

# transfer function
# manual definition of the transfer function to obtain "x-ray" effect
def transfer_func_ct(vals, binary=False, cathlab=False):
    new_vals = copy.deepcopy(vals).astype('float')

    x1 = 0
    x2 = 753
    x3 = 1585.85
    x4 = 2332.9
    x5 = 3306.18
    x6 = 4000
    
    # binary
    if binary:
        y1 = 0
        y2 = 0
        y3 = 0 #0.0001 #0.01
        y4 = 0
        y5 = 0.2#0
        y6 = 0.4
    # "disappearing" vessels
    else:
        # used for ALL experiments
        y1 = 0
        y2 = 0
        y3 = 0.05  #0.02
        y4 = 0
        y5 = 0.2 #0.2
        y6 = 0.4

    new_vals[new_vals < x1] = y1

    new_vals[(new_vals >= x1) & (new_vals < x2)] = line(new_vals, [x1, y1], [x2, y2])
    new_vals[(new_vals >= x2) & (new_vals < x3)] = line(new_vals, [x2, y2], [x3, y3])
    new_vals[(new_vals >= x3) & (new_vals < x4)] = line(new_vals, [x3, y3], [x4, y4])
    new_vals[(new_vals >= x4) & (new_vals < x5)] = line(new_vals, [x4, y4], [x5, y5])
    new_vals[(new_vals >= x5) & (new_vals < x6)] = line(new_vals, [x5, y5], [x6, y6])
    new_vals[new_vals >= x6] = y6

    return new_vals

def get_interpolator_from_vol_sdf(vol_grid, vol_scale=1):
    transl_grid = vol_grid.scale(vol_scale, inplace=False)
    transl_grid = transl_grid.translate(-np.array(transl_grid.center_of_mass()), inplace=False)

    grid_bounds = transl_grid.bounds
    dimensions = transl_grid.dimensions

    # scales the volume based on its bounds to be [-1, 1]
    # we can do this because we work with "bounded" scenes
    # grid_scaling_factor = np.max(grid_bounds)

    points = transl_grid.points
    # round points to 3 decimals to minimize memory
    points = points.round(decimals=3)

    points_x = np.unique(points[:,0])
    points_y = np.unique(points[:,1])
    points_z = np.unique(points[:,2])

    scalars = transl_grid.point_data['scalars'].astype('float')
    # transfer function
    scalars = rev_sigmoid(scalars, c1=2)

    scalars_dim = scalars.reshape(dimensions)
    fill_value = np.min(scalars_dim)

    interpolator = RegularGridInterpolator((points_x, points_y, points_z), scalars_dim, method='linear', bounds_error=False, fill_value=fill_value)

    return interpolator, grid_bounds

def get_interpolator_from_vol_ct(vol_grid, data_folder_name, translation=np.array([0,0,0]), transfer_func=None, binary=False, cathlab=False):
    # rotating for cathlab positioning
    if cathlab:
        rotate_grid = vol_grid.rotate_vector(vector=(1,0,0), angle = -90, point = vol_grid.center,inplace=False)
    else:
        rotate_grid = vol_grid
    # small manual translation to have LCA in center of view
    transl_grid = rotate_grid.translate(-np.array(vol_grid.center)+translation, inplace=False)

    grid_bounds = transl_grid.bounds
    dimensions = transl_grid.dimensions

    # scales the volume based on its bounds to be [-1, 1]
    # we can do this because we work with "bounded" scenes
    grid_scaling_factor = np.max(grid_bounds)

    interpolator, _, scalars = get_interpolator_from_grid(transl_grid, transfer_func, binary)
    
    # save vtk with new transfer function
    transl_grid.point_data['scalars'] = scalars
    # rotate and scale so it matches prediction volume
    transl_grid.rotate_x(-90, inplace=True)

    binary_str = 'binary' if binary else ''
    transl_grid.save(f'{data_folder_name}transferfunc{binary_str}.vtk', binary=binary)

    return interpolator, grid_bounds

def get_interpolator_from_grid(grid, transfer_func=None, binary=False):
    scalars = grid.point_data['scalars'].astype('float')
    points = grid.points

    if transfer_func:
        scalars = transfer_func(scalars, binary=binary)
    # round points to 3 decimals to minimize memory
    points = points.round(decimals=3)

    points_x = np.unique(points[:,0])
    points_y = np.unique(points[:,1])
    points_z = np.unique(points[:,2])

    # the interpolators require a certain format of points and scalars
    # so we use a kd-tree to query them (efficient)
    kd_tree = KDTree(points)
    xg, yg, zg = np.meshgrid(points_x, points_y, points_z, indexing='ij')
    _, indices = kd_tree.query(np.concatenate((np.expand_dims(xg, axis=-1), np.expand_dims(yg, axis=-1), np.expand_dims(zg, axis=-1)), axis=-1))
    scalars_dim = scalars[indices]

    fill_value = np.min(scalars_dim)

    interpolator = RegularGridInterpolator((points_x, points_y, points_z), scalars_dim, method='linear', bounds_error=False, fill_value=fill_value)

    return interpolator, points, scalars

def get_ray_values(theta, phi, larm, src_pt, img_width, img_height, focal_length, device, translation=np.array([0,0,0])):
    # obtain rotation matrix based on angles
    src_matrix = source_matrix(src_pt, theta, phi, larm, translation)
    tform_cam2world = torch.from_numpy(src_matrix).to(device)

    # do point & ray sampling
    ii, jj = torch.meshgrid(
        torch.arange(0, img_width).to(tform_cam2world),
        torch.arange(0, img_height).to(tform_cam2world),
        indexing='xy'
    )

    directions = torch.stack([(ii - img_width / 2) / focal_length,
                        -(jj - img_height / 2) / focal_length,
                        -torch.ones_like(ii)
                        ], dim=-1)

    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1).to(device)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape).to(device) 
    return ray_origins, ray_directions, src_matrix, ii, jj

def get_depth_values(near_thresh, far_thresh, depth_samples_per_ray, device, stratified=True):
    t_vals = torch.linspace(0., 1., depth_samples_per_ray)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)

    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)

    # stratified samples in those intervals
    if stratified:
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(device)
    return depth_values

def ray_tracing(interpolator, angles, ray_origins, ray_directions, depth_values, img_width, img_height, ii, jj, batch_size, device, proj_folder_name, type='ct', invert=False):
    img = torch.ones((int(np.ceil(img_height)), int(np.ceil(img_width))))
    # loop over image (because it has high memory consumption)
    for i_index in range(0, ii.shape[0], batch_size):
        for j_index in range(0, jj.shape[0], batch_size):

            query_points = ray_origins[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size] + ray_directions[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size] * depth_values[..., :, None]

            one_e_10 = torch.tensor([1e10], dtype=depth_values.dtype, device=depth_values.device)
            dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

            ray_points = query_points.cpu().numpy().reshape((-1, 3))

            interp_vals = torch.from_numpy(interpolator(ray_points)).to(device).reshape(query_points.shape[:-1])

            # x-ray image
            if type == 'ct':
                norm_dists = dists * torch.norm(ray_directions[..., None, :][i_index:i_index+batch_size, j_index:j_index+batch_size], dim=-1)
                
                weights = torch.exp(-interp_vals*norm_dists)
            else:
                weights = torch.exp(-interp_vals)

            img_val = torch.prod(weights, dim=-1)
            img[i_index:i_index+batch_size,j_index:j_index+batch_size] = img_val

        try:
            plt.imsave(f'{proj_folder_name}image-{angles[0]}-{angles[1]}-{angles[2]}.png', img.numpy(), cmap='gray', vmin=0, vmax=1)
        except Exception as e: 
            print(e)
            print('error saving image')

    return img

def get_weighted_img(img, frangi_alpha, frangi_beta, theta, phi, larm, proj_folder_name, sampling_strategy='frangi', invert=False):
    if sampling_strategy == 'frangi':
        img_binary = frangi(img, alpha=frangi_alpha, beta=frangi_beta)
    else:
        img_binary = np.zeros(img.shape)
        img_binary[img < 1] = 1
    # normalize
    img_binary -= np.min(img_binary)
    img_binary /= np.max(img_binary)

    # compute euclidean distance transform
    img_transf = distance_transform_edt(img_binary)

    # # normalize
    img_transf -= np.min(img_transf)
    img_transf /= np.max(img_transf)

    # to ensure that there are enough "non-zero" entries for weight sampling
    img_transf += 1e-10
    plt.imsave(f'{proj_folder_name}image-transform-{theta}-{phi}-{larm}.png', img_transf)

    return img_transf

def visualize_volume(grid_bounds, grid_scaling_factor=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # visualize volume
    x_bounds = grid_bounds[0:2]
    y_bounds = grid_bounds[2:4]
    z_bounds = grid_bounds[4:6]
    for x in x_bounds:
        for y in y_bounds:
            for z in z_bounds:
                ax.scatter(x / grid_scaling_factor, y / grid_scaling_factor, z / grid_scaling_factor, color='red')
                ax.scatter(x, y, z, color='grey')
    plt.show()

def visualize_query_points(x_pixs, y_pixs, ray_origins, ray_directions, depth_values, grid_scaling_factor=1):
    x_pixs = [0, img_width//2-1, img_width-1]
    y_pixs = [0, img_height//2-1, img_height-1]
    for x in x_pixs:
        for y in y_pixs:
            query_points = (ray_origins[..., None, :][y, x] + ray_directions[..., None, :][y, x] * depth_values[..., :, None]).cpu().numpy()

            point1 = query_points[0]
            point2 = query_points[-1]
            points = np.array([point1, point2]).T

            points_scale = points / grid_scaling_factor
            ax.plot(points[0, :], points[1, :], points[2, :], c='grey')
            ax.plot(points_scale[0, :], points_scale[1, :], points_scale[2, :], c='red')
    plt.show()