import torch 
import pyvista as pv
import numpy as np
import itertools
import os
import pandas as pd
import argparse
import ast
import pdb
import matplotlib.pyplot as plt

from helpers import ray_tracing, get_interpolator_from_vol_ct, transfer_func_ct, get_ray_values, get_depth_values, get_weighted_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--limited_size', help='Angle range to sample the projections in')
parser.add_argument('--number_angles', help='Number of projections to sample per axis')
parser.add_argument('--center_point', help='Center point for the angle sampling')
parser.add_argument('--binary', help='Whether images are binary or not')
parser.add_argument('--sampling_strategy', help='What sampling strategy to use, options: frangi, segmentation or random')

args = parser.parse_args()
arg_dict = vars(args)

# define general parameters
limited_size = float(arg_dict['limited_size']) if arg_dict['limited_size'] is not None else 360
number_angles = float(arg_dict['number_angles']) if arg_dict['number_angles'] is not None else 72 #(+1) e.g. 5 + 1 = 6
center_point = ast.literal_eval(arg_dict['center_point']) if arg_dict['center_point'] is not None else [90,0] #[0,90] #[90,0]
binary = arg_dict['binary'] == 'True' if arg_dict['binary'] is not None else False
sampling_strategy = arg_dict['sampling_strategy'] if arg_dict['sampling_strategy'] is not None else 'frangi'

unseen = False

custom_angle = [[135, 135]]

# camera optimization parameters w.r.t. grid size for translations
min_shift_translation = 0
max_shift_translation = 0
# camera optimization parameters w.r.t. % for rotation
min_shift_rotation = 0
max_shift_rotation = 0

file_name = f'clinical-angles'

batch_size = 100 # for looping over image
grid_scaling_factor = 1

if sampling_strategy == 'frangi':
    frangi_alpha = 12 if binary else 0.5
    frangi_beta = 0.5 if binary else 0.5

# small manual translation to have LCA in center of view
manual_translation = np.array([-30, 10, -30])

# define camera parameters
focal_length = 1300
src_pt = np.array([0, 0, focal_length+200])
sample_outside = 210 #175
near_thresh = src_pt[2] - sample_outside#-960
far_thresh = src_pt[2] + sample_outside#-620
depth_samples_per_ray = sample_outside*2

# define image width and height
img_width = 100
img_height = 100
nb_pixels = img_height * img_width
max_img_dim = max(img_width, img_height)

# projection angles
if number_angles > 0:
    angle_step = limited_size/number_angles
binary_str = 'binary' if binary else ''

# angles
larm = 0

# maintain original starting rotation for theta
theta_rot = 0
phi_rot = 0

if center_point[0] > 0:
    theta_rot += center_point[0]
if center_point[1] > 0:
    phi_rot += center_point[1]

if number_angles > 0:
    # th_angles = np.array([0, 30])
    # ph_angles = np.array([0])
    th_angles = (np.arange(-limited_size//2, limited_size//2+1, angle_step) + theta_rot)
    ph_angles = (np.arange(-limited_size//2, limited_size//2+1, angle_step) + phi_rot)

    # convert to [-180, 180]
    th_angles[th_angles > 180] = th_angles[th_angles > 180] - 180
    ph_angles[ph_angles > 180] = ph_angles[ph_angles > 180] - 180

    # th_angles = np.array([0, 20,  0, 10, -50, -50,   0]) 
    # ph_angles = np.array([0, -20, 30, 40,  30, -30, -30])
    th_ph_angles = np.array([np.array(val) for val in itertools.product(th_angles, ph_angles)])
else:
    th_ph_angles = np.array([[90, 0], [0,90]])

# we define one custom angle for testing
th_ph_angles = np.append(th_ph_angles, custom_angle, axis=0)

if unseen:
    th_angles_unseen = np.arange(-180, 180, 18)
    ph_angles_unseen = np.arange(-180, 180, 18)
    th_ph_angles_unseen = np.array([np.array(val) for val in itertools.product(th_angles_unseen, ph_angles_unseen)])

# load the data
data_folder_name = 'data/ct/'

# this vtk file is a preprocessed ct volume with already enhanced constrast in the arteries.
# we use the additional transfer function to obtain the synthetic angio's
proc_file = 'processed-new.vtk'

# check if proj folder exists
proj_folder_name = f'{data_folder_name}/'
if not os.path.isdir(proj_folder_name):
    os.mkdir(proj_folder_name)

# read vtk file
vol_reader = pv.get_reader(data_folder_name + proc_file)
vol_grid = vol_reader.read()

interpolator, grid_bounds = get_interpolator_from_vol_ct(vol_grid, data_folder_name, manual_translation, transfer_func=transfer_func_ct, binary=binary)

if sampling_strategy == 'segmentation':
    binary_interpolator, _ = get_interpolator_from_vol_ct(vol_grid, data_folder_name, manual_translation, transfer_func=transfer_func_ct, binary=True)

# save query pts as vtk file
# obtain 3D point cloud
outside = 75
near_vol_thresh = -outside
far_vol_thresh = outside
t = np.linspace(near_vol_thresh, far_vol_thresh, 200)

mesh_grid = np.meshgrid(t, t, t)
query_pts = torch.Tensor(np.stack(mesh_grid, -1)).to(device)

ray_points = query_pts.cpu().numpy().reshape((-1, 3))
interp_vals = torch.from_numpy(interpolator(ray_points)).to(device)#.reshape(query_pts.shape[:-1])

grid = pv.StructuredGrid(mesh_grid[0], mesh_grid[1], mesh_grid[2])
grid.point_data['scalars'] = interp_vals.cpu().numpy()
grid.save(f"{proj_folder_name}ground-truth.vtk")

# 100% is shifted half size of volume (max dim)
max_grid_dim = max(np.abs(grid_bounds))

# values to store cttoproj
image_ids = []
thetas = []
theta_shifts = []
phis = []
phi_shifts = []
larms = []
larm_shifts = []
translations_x = []
translations_y = []
translations_z = []
matrices = []
unshifted_matrices = []
images = []
dist_images = []
img_widths = []
img_heights = []
focal_lengths = []
near_threshs = []
far_threshs = []
depth_samples = []
depth_values_lst = []

# values to store projtoray
image_ids_rays = []
x_positions = []
y_positions = []
ray_origins_x = []
ray_origins_y = []
ray_origins_z = []
ray_directions_x = []
ray_directions_y = []
ray_directions_z = []
pixel_values = []
distance_pixel_values = []

for i, angles in enumerate(th_ph_angles):
    theta, phi = angles
    image_id = f'{theta}-{phi}'.replace('.', ',')

    # random angular shift
    theta_shift, phi_shift, larm_shift = np.random.uniform(low=-max_shift_rotation, high=max_shift_rotation, size=3)
    # random number image width & image height 10% 20% etc.
    translation = np.random.uniform(low=-max_shift_translation, high=max_shift_translation, size=3) * max_grid_dim

    # including the shift translation & rotation (src_matrix for images generated)
    if i < len(th_ph_angles) - 1:
        ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values(theta+theta_shift, phi+phi_shift, larm+larm_shift, src_pt, img_width, img_height, focal_length, device, translation)
    else:
        ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values(theta, phi, larm, src_pt, img_width, img_height, focal_length, device)
    depth_values = get_depth_values(near_thresh, far_thresh, depth_samples_per_ray, device)

    # excluding the shift translation & rotation
    _, _, unshifted_src_matrix, _, _ = get_ray_values(theta, phi, larm, src_pt, img_width, img_height, focal_length, device)

    img = ray_tracing(interpolator, [theta,phi,larm], ray_origins, ray_directions, depth_values, img_width, img_height, ii, jj, batch_size, device, proj_folder_name)
    
    img_to_transf = np.array(img)
    if not binary:
        quantile = np.percentile(img_to_transf, 10)
        img_to_transf[img_to_transf > quantile] = 1

    if sampling_strategy == 'frangi':
        img_transf = get_weighted_img(img_to_transf, frangi_alpha, frangi_beta, theta, phi, larm, proj_folder_name, sampling_strategy)
    elif sampling_strategy == 'segmentation':
        binary_img = ray_tracing(binary_interpolator, [theta,phi,larm], ray_origins, ray_directions, depth_values, img_width, img_height, ii, jj, batch_size, device, proj_folder_name)
        img_transf = get_weighted_img(binary_img, None, None, theta, phi, larm, proj_folder_name, sampling_strategy)
    else:
        img_transf = np.ones(img_to_transf.shape)

    # set dataframe values for cttoproj
    image_ids.append(image_id)
    thetas.append(theta)
    phis.append(phi)
    larms.append(larm)
    theta_shifts.append(theta_shift)
    phi_shifts.append(phi_shift)
    larm_shifts.append(larm_shift)
    translations_x.append(translation[0])
    translations_y.append(translation[1])
    translations_z.append(translation[2])
    matrices.append(src_matrix.tolist())
    unshifted_matrices.append(unshifted_src_matrix.tolist())
    images.append(img.tolist())
    dist_images.append(img_transf.tolist())
    img_widths.append(img_width)
    img_heights.append(img_height)
    focal_lengths.append(focal_length)
    near_threshs.append(near_thresh)
    far_threshs.append(far_thresh)
    depth_samples.append(depth_samples_per_ray)
    depth_values_lst.append(depth_values.tolist())

    # set dataframe values for projtoray
    image_ids_rays.append(np.repeat(image_id, nb_pixels))
    x_positions.append(ii.flatten().cpu().numpy())
    y_positions.append(jj.flatten().cpu().numpy())

    pixel_values.append(img.numpy().flatten())
    distance_pixel_values.append(img_transf.flatten())

    ray_origins = ray_origins.reshape((-1, 3)).cpu().numpy()
    ray_origins_x.append(ray_origins[:,0])
    ray_origins_y.append(ray_origins[:,1])
    ray_origins_z.append(ray_origins[:,2])

    ray_directions = ray_directions.reshape((-1,3)).cpu().numpy()
    ray_directions_x.append(ray_directions[:,0])
    ray_directions_y.append(ray_directions[:,1])
    ray_directions_z.append(ray_directions[:,2])

# normalize imgs
images -= np.min(images)
images /= np.max(images)
images = images.tolist()

print('creating df')

df = pd.DataFrame({'image_id': image_ids, 'theta': thetas, 'phi': phis, 'larm': larms, 
'theta_shift': theta_shifts, 'phi_shift': phi_shifts, 'larm_shift': larm_shifts,
'translation_x': translations_x, 'translation_y': translations_y, 'translation_z': translations_z, 
'tform_cam2world': matrices, 'unshifted_tform_cam2world': unshifted_matrices, 'image_data': images, 'image_distance_data': dist_images,
'org_img_width': img_widths, 'org_img_height': img_heights, 'focal_length': focal_lengths, 'near_thresh': near_thresh, 
'far_thresh': far_threshs, 'depth_sample': depth_samples, 'grid_scaling_factor': grid_scaling_factor, 'depth_values': depth_values_lst, 'src_pt_z': src_pt[2] })

print('created df')

# to list
df['image_data'] = df['image_data'].map(list)
df['image_distance_data'] = df['image_distance_data'].map(list)
df['tform_cam2world'] = df['tform_cam2world'].map(list)
df['depth_values'] = df['depth_values'].map(list)

df.to_csv(f'{data_folder_name}/df-{file_name}-{binary_str}-cttoproj.csv', sep=';')
print('saved df')

image_ids_rays = np.array(image_ids_rays).reshape(-1).tolist()
pixel_values = np.array(pixel_values).reshape(-1).tolist()
distance_pixel_values = np.array(distance_pixel_values).reshape(-1).tolist()
x_positions = np.array(x_positions).reshape(-1).tolist()
y_positions = np.array(y_positions).reshape(-1).tolist()

ray_origins_x = np.array(ray_origins_x).reshape(-1).tolist()
ray_origins_y = np.array(ray_origins_y).reshape(-1).tolist()
ray_origins_z = np.array(ray_origins_z).reshape(-1).tolist()

ray_directions_x = np.array(ray_directions_x).reshape(-1).tolist()
ray_directions_y = np.array(ray_directions_y).reshape(-1).tolist()
ray_directions_z = np.array(ray_directions_z).reshape(-1).tolist()

df = pd.DataFrame({'image_id': image_ids_rays, 'pixel_value': pixel_values, 'distance_pixel_value': distance_pixel_values,
    'x_position': x_positions, 'y_position': y_positions, 
    'ray_origins_x': ray_origins_x, 'ray_origins_y': ray_origins_y, 'ray_origins_z': ray_origins_z,
    'ray_directions_x': ray_directions_x, 'ray_directions_y': ray_directions_y, 'ray_directions_z': ray_directions_z })

df.to_csv(data_folder_name + '/' + f'df-rays-{file_name}-{binary_str}-{img_height}.csv', sep=';')