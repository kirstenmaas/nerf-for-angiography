import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pdb

def get_cmap(cmap_base, n=100, vmin=0.2, vmax=1):
    cmap = plt.get_cmap(cmap_base)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax),
        cmap(np.linspace(vmin, vmax, n)))
    return cmap


data_name = 'ct'
binary = False

sampling = 'Random'
# sampling = ''

folder_loc = f'cases/{data_name}/'

x_variable = 'Limited projections'

y_variables = ['DICE 2D mean']

# std_variable = 'PSNR std'
std_variable = None

font_size = 90
label_font_size = 80

group_variable = 'Sparse projections' # 'Sparse projections'
# group_variable = 'Model architecture'
center_point = '(90 0)'
# center_point = ''

name = f'{x_variable.lower()}'
for y_variable in y_variables:
    name += f' - {y_variable.lower()}'

trend_line = False

# limited_projection_angles = [30, 60, 90, 180]
limited_projection_angles = [5, 15, 30, 45, 60, 90, 180]
# label_unique_x_variables = [5, 30, 60, 90, 180]
label_unique_x_variables = [30, 60, 90, 180]
# limited_projection_angles = [5, 15, 30]

vmin, vmax = 0, 1

cmap_base = 'viridis'
if group_variable == 'Sparse projections':
    cmap_base = 'inferno'

PSNR_max = 47.8239

cmap = get_cmap(cmap_base)

filters = [ 
    { "property": 'Data', "select": { "equals": data_name.upper() }}, 
    { "property": 'Binary', "checkbox": { "equals": binary }},
]

if len(sampling) > 0:
    filters.append({ "property": 'Sampling', "multi_select": { 'contains': sampling + ' sampling' } })
    name += f' - {sampling}'

if len(center_point) > 0:
    filters.append({ "property": 'Centerpoint', "select": { "equals": center_point }})
    name += f' - {center_point}'

if group_variable != 'Model architecture':
    filters.append({ "property": 'Model architecture', "select": { "equals": '4x128' }})

if std_variable:
    name += f' - std={std_variable}'

if not binary:
    name += '- background'

# load the data with results (removed from original code)
df = None
df = df[((df['Sparse projections'] != 4) | (df['Limited projections'] != 180))]

fig = plt.figure(figsize=(20,15))
# rc('font',**{'family':'serif','serif':['Roboto']})

if group_variable:
    name += f' by {group_variable.lower()}'

    # remove non-interesting limited projections
    df = df[df['Limited projections'].isin(limited_projection_angles)]
    df = df[df['Sparse projections'] > 1]

    # only one sampling technique to visualize
    if len(sampling) > 0:
        df = df[df['Sampling'].str.contains(sampling)]
    
    df['LPIPS mean'] = 1 - df['LPIPS mean']
    df['DISTS mean'] = 1 - df['DISTS mean']
    
    df = df.sort_values(by=[group_variable])
    unique_group = pd.unique(df[group_variable])

    norm = mpl.colors.Normalize(vmin=0, vmax=len(unique_group))
    colors = cm.ScalarMappable(norm, cmap=cmap)

    for idx, group in enumerate(unique_group):
        rows = df[df[group_variable] == group].sort_values(by=[x_variable])#.groupby(group_variable)

        # visualize multiple sampling techniques in one plot
        if len(sampling) == 0:
            unique_sampling = pd.unique(df['Sampling'])
            for jdx, sample_group in enumerate(unique_sampling):
                sample_rows = rows[rows['Sampling'] == sample_group]


                label = f'{str(int(group))} {group_variable.lower()}'
                if group_variable == 'Limited projections':
                    label = f'{str(int(group))} {group_variable.lower()}'
                    label += '°'

                if group_variable != 'Model architecture':
                    colors = None
                    if 'Frangi' in sample_group:
                        colors = cmaps[0]
                    elif 'Segmentation' in sample_group:
                        colors = cmaps[1]
                    elif 'Random' in sample_group:
                        colors = cmaps[1]
                
                if type(colors) is tuple:
                    plt.plot(sample_rows[x_variable], sample_rows[y_variables[0]], c=colors[idx], label=label, linewidth=10.0)
                else:
                    plt.plot(sample_rows[x_variable], sample_rows[y_variables[0]], c=colors.to_rgba(idx), label=label, linewidth=10.0)

        else:
            colors = cm.ScalarMappable(norm, cmap=cmap_base)
            for c, y_variable in enumerate(y_variables):
                
                variables = [x_variable, y_variable]
                if std_variable:
                    variables.append(std_variable)

                means = rows[variables].groupby(x_variable).mean(numeric_only=True)
                stds = rows[variables].groupby(x_variable).std()

                label = f'{str(int(group))}'
                if group_variable == 'Limited projections':
                        label = f'{str(int(group))}'
                        label += '°'
                label += f' {group_variable.lower()}'

                if type(colors) is tuple:
                    plt.plot(means.index, means[y_variable], c=colors[idx], label=label, linewidth=10)       
                else:
                    plt.plot(means.index, means[y_variable], c=colors.to_rgba(idx), label=label, linewidth=10)


plt.xlabel(x_variable, fontsize=font_size, c='white')
plt.ylabel(y_variable, fontsize=font_size, c='white')

# ticks
unique_x_variables = pd.unique(df[x_variable])
if x_variable == 'Limited projections':
    unique_x_variables = label_unique_x_variables

if len(unique_x_variables) < 10:
    labels = unique_x_variables
    if x_variable == 'Limited projections':
        labels = [str(int(label)) + '°' for label in labels]
    elif x_variable == 'Sparse projections':
        labels = [str(int(label)) for label in labels]
    print(labels)
    plt.xticks(unique_x_variables, labels=labels, fontsize=font_size)

plt.yticks(fontsize=font_size)

if 'PSNR' in y_variable:
    plt.yticks([5, 15, 25, 35, 45], fontsize=font_size)
    plt.ylim(5, 48)
    print('PSNR')
elif 'SSIM' in y_variable:
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=font_size)
    plt.ylim(0.1, 1)
    print('SSIM')
elif 'DICE 2D' in y_variable:
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=font_size)
    plt.ylim(0, 1)
    print('DICE 2D')
elif 'DICE 3D' in y_variable:
    plt.ylim(0, 1)
    print('DICE 3D')
elif 'LPIPS' in y_variable:
    plt.ylim(0, 1)
else:
    plt.ylim(0.1, 1)

# plt.legend(fontsize=60, loc='lower right')
plt.savefig(f'{folder_loc}{name}.png', bbox_inches='tight')