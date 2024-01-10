import torch 
import pyvista as pv
import numpy as np
import itertools
import os
import pandas as pd
import skimage as sk
import pdb
import matplotlib.pyplot as plt
import copy

from helpers import ray_tracing, get_interpolator_from_vol_sdf, visualize_volume, get_ray_values, get_depth_values, get_weighted_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define general parameters
limited_size = 25
number_angles = 4
# custom_angle = [[135,135]]
custom_angle = [[112.5, 112.5]]
batch_size = 100
grid_scaling_factor = 1/8

apply_frangi = False

frangi_alpha = 12
frangi_beta = 0.5

file_name = f'sparse-{number_angles}'

# define camera parameters
focal_length = 4000
src_pt = np.array([0, 0, focal_length])
sample_outside = 1000
near_thresh = src_pt[2] - sample_outside#-960
far_thresh = src_pt[2] + sample_outside#-620
depth_samples_per_ray = sample_outside*2

# define image width and height
img_width = int(1200*grid_scaling_factor)
img_height = int(1300*grid_scaling_factor)

new_img_height = img_height
new_img_width = int((img_width/img_height)*new_img_height)
nb_pixels = new_img_height * new_img_width

# projection angles
angle_step = limited_size/number_angles

# angles
larm = 0
th_angles = np.arange(0, limited_size+1, angle_step)
ph_angles = np.arange(0, limited_size+1, angle_step)
th_ph_angles = np.array([np.array(val) for val in itertools.product(th_angles, ph_angles)])

# we define one custom angle for testing
th_ph_angles = np.append(th_ph_angles, custom_angle, axis=0)

# load the data
data_folder_name = 'data/stl/'
data_name = 'LCA'
folder_name = 'sdftoproj'
sdf_file = 'SDF-LCA.vtk'


# check if proj folder exists
proj_folder_name = f'{data_folder_name}{data_name}/{folder_name}/'
if not os.path.isdir(proj_folder_name):
    os.mkdir(proj_folder_name)

vol_reader = pv.get_reader(data_folder_name + sdf_file)#pv.get_reader(volume_file) #pv.get_reader(segmentation_file)
vol_grid = vol_reader.read()
interpolator, grid_bounds = get_interpolator_from_vol_sdf(vol_grid, grid_scaling_factor)

# visualize_volume(grid_bounds)

# values to store sdftoproj
image_ids = []
thetas = []
phis = []
larms = []
translations_x = []
translations_y = []
translations_z = []
matrices = []
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

for angles in th_ph_angles:
    theta, phi = angles
    image_id = f'{theta}-{phi}'.replace('.', ',')
    print(image_id)

    translation = np.array([0,0,0])
    ray_origins, ray_directions, src_matrix, ii, jj = get_ray_values(theta, phi, larm, src_pt, img_width, img_height, focal_length, device, translation)
    depth_values = get_depth_values(near_thresh, far_thresh, depth_samples_per_ray, device)

    # x_pixs = [0, img_width//2-1, img_width-1]
    # y_pixs = [0, img_height//2-1, img_height-1]
    # visualize_query_points(x_pixs, y_pixs, ray_origins, ray_directions, depth_values, grid_scaling_factor=1)

    img = ray_tracing(interpolator, [theta,phi,larm], ray_origins, ray_directions, depth_values, img_width, img_height, ii, jj, batch_size, device, proj_folder_name, type='sdf')

    # normalize img
    img -= torch.min(img)
    img /= torch.max(img)

    plt.imsave(f'{proj_folder_name}image-{theta}-{phi}-{larm}.png', img.numpy(), cmap='gray', vmin=0, vmax=1)

    img_transf = get_weighted_img(img, frangi_alpha if apply_frangi else None, frangi_beta if apply_frangi else None, theta, phi, larm, proj_folder_name)

    resized_img = sk.transform.resize(img, (int(new_img_height), int(new_img_width)))
    resized_img_transf = sk.transform.resize(img_transf, (int(new_img_height), int(new_img_width)))

    # set dataframe values for sdftoray
    image_ids.append(image_id)
    thetas.append(theta)
    phis.append(phi)
    larms.append(larm)
    translations_x.append(translation[0])
    translations_y.append(translation[1])
    translations_z.append(translation[2])
    matrices.append(src_matrix.tolist())
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

    pixel_values.append(resized_img.flatten())
    distance_pixel_values.append(resized_img_transf.flatten())

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

df = pd.DataFrame({'image_id': image_ids, 'theta': thetas, 'phi': phis, 'larm': larms, 'translation_x': translations_x, 'translation_y': translations_y,
'translation_z': translations_z, 'tform_cam2world': matrices, 'image_data': images, 'image_distance_data': dist_images,
'org_img_width': img_widths, 'org_img_height': img_heights, 'focal_length': focal_lengths, 'near_thresh': near_thresh, 
'far_thresh': far_threshs, 'depth_sample': depth_samples, 'depth_values': depth_values_lst, 'src_pt_z': src_pt[2] })

# to list
df['image_data'] = df['image_data'].map(list)
df['image_distance_data'] = df['image_distance_data'].map(list)
df['tform_cam2world'] = df['tform_cam2world'].map(list)
df['depth_values'] = df['depth_values'].map(list)

df.to_csv(f'{data_folder_name}{data_name}/df-{file_name}-sdftoproj.csv', sep=';')

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
df.to_csv(data_folder_name + data_name + '/' + f'df-rays-{file_name}-{new_img_height}.csv', sep=';')