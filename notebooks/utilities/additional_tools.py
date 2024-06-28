import logging, copy, re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation
    
def interpolate_volume(origVol, origCoords, intCoords, intMethod='linear', 
                       boundError=True, fillValue=0, getFunction=False):
    """Interpolate the volume `origVol` with coordinates `origCoords` to the grid\n
    defined by the coordinates `intCoords`. Note that the original grid coordinates must be regular.

    Parameters:
    -----------
        `origVol` (ndarray): Original volume
        `origCoords` (list or typle): coordinates of original volume ordered as s, v, h
        `intCoords` (list or tuple): coordinates of desired output grid given as s, v, h
        `intMethod` (str, optional): Interpolation method. Defaults to 'linear'.
        `boundError` (bool, optional): Flag used by `RegulatGridInterpolator` to activate a raised 
                                     error when values outside of the range of the interpolated
                                     function are found. Defaults to True.
        `getFunction` (bool, optional): Flag to return interpolating function. Defaults to False.

    Returns:
    --------
        `ndarray`: Interpolated volume
        `RegularGridInterpolator object` (ndarray, optional): Interpolating function
    """    
    
    # grab input coordinates
    s, v, h = origCoords
    sInt, vInt, hInt = intCoords
    
    # create interpolating function
    interpFunc = RegularGridInterpolator((s, v, h), origVol, method = intMethod,
                                         bounds_error=boundError, fill_value=fillValue)
    
    # create grid where values will be interpolated
    ss, vv, hh  = np.meshgrid(sInt, vInt, hInt, indexing='ij')
    coords = np.stack((ss.flat, vv.flat, hh.flat), axis=1)
    
    # find interpolated values at new grid points
    intVol = interpFunc(coords).reshape([len(sInt),len(vInt), len(hInt)])
    
    if not getFunction: return intVol
    
    return interpFunc, intVol

def patchify(data, d, s):
    
    d = np.array(d)
    s = np.array(s)

    D = data.shape

    # number of patches to use for each dimension
    N = np.ceil(np.divide(D - d, s)).astype(int) 

    # Number of extra voxels needed for padding
    m = np.abs(D - (np.multiply(N,s) + d)).astype(int)

    # Pad width along each dimension
    pad_width = tuple([(m[n]//2,m[n]-m[n]//2) for n in range(len(D))])

    # pad input data
    padded_data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

    # update the number of patches 
    N += 1
    
    if len(D) == 2: # 2D

        tlc = np.array([[r*s[0], c*s[1]] for r in range(N[0]) for c in range(N[1])]) # top left corner
        brc = np.array([[r*s[0]+d[0], c*s[1]+d[1]] for r in range(N[0]) for c in range(N[1])]) # bottom right corner
        # create patches 
        patches = {n:{'data':padded_data[tlc[n][0]:brc[n][0], tlc[n][1]:brc[n][1]]} for n in range(np.prod(N).astype(int))}

    if len(D) == 3: # 3D

        tlc = np.array([[z*s[0], r*s[1], c*s[2]] for z in range(N[0]) for r in range(N[1]) for c in range(N[2])])# top left corner
        brc = np.array([[z*s[0]+d[0], r*s[1]+d[1], c*s[2]+d[2]] for z in range(N[0]) for r in range(N[1]) for c in range(N[2])]) # bottom right corner
        # create patches 
        patches = {'data':{n:padded_data[tlc[n][0]:brc[n][0], tlc[n][1]:brc[n][1], tlc[n][2]:brc[n][2]] for n in range(np.prod(N).astype(int))}}
         
    patches['top_left_corner'] = tlc
    patches['bottom_right_corner'] = brc
    patches['pad_width'] = pad_width
    patches['original_shape'] = data.shape
    patches['padded_shape'] = padded_data.shape

    return patches

def unpatchify(patches, rim_crop = 0):

    # grab values from patches dictionary    
    tlc = [n + rim_crop for n in patches['top_left_corner']]
    brc = [n - rim_crop for n in patches['bottom_right_corner']]
    pw = patches['pad_width'] 
    o_shape = patches['original_shape'] 
    p_shape = patches['padded_shape']
    
    # Initialize variables
    merged = np.zeros(p_shape)
    temp = np.ones(p_shape)
    weights = np.zeros(p_shape)

    if len(o_shape) == 3: # 3D

        for n in patches['data'].keys():
            fi, li = rim_crop, -rim_crop if rim_crop > 0 else None

            merged[tlc[n][0]:brc[n][0], tlc[n][1]:brc[n][1], tlc[n][2]:brc[n][2]] += patches['data'][n][fi:li, fi:li, fi:li]
            weights[tlc[n][0]:brc[n][0], tlc[n][1]:brc[n][1], tlc[n][2]:brc[n][2]] += temp[tlc[n][0]:brc[n][0], tlc[n][1]:brc[n][1], tlc[n][2]:brc[n][2]]

    merged = np.divide(merged, weights, where= weights>0)
    
    # remove padding
    pw = [(p[0], -p[1] if p[1] != 0 else None) for p in pw]    
    reconstructed = merged[pw[0][0]:pw[0][-1],pw[1][0]:pw[1][-1],pw[2][0]:pw[2][-1]]
    
    return reconstructed

def rotate_coordinates(coord_mat, theta_c, theta_g, precision = 4):
        
    if len(coord_mat.shape) == 1: 
        coord_mat = np.expand_dims(coord_mat, axis = 0).T
        single_point = True

    # Gantry Rotation: rotation about z-axis
    rot_mat_z = np.zeros([3,3])
    rot_mat_z[0,0], rot_mat_z[0,1] = np.cos(np.radians(-theta_g)), np.sin(np.radians(-theta_g))
    rot_mat_z[1,0], rot_mat_z[1,1] = -np.sin(np.radians(-theta_g)), np.cos(np.radians(-theta_g))
    rot_mat_z[2,2] = 1

    ## rotate coordiantes
    rot_coords_gantry = np.dot(rot_mat_z, coord_mat)

    # Couch Rotation: rotation about the y-axis
    rot_mat_y = np.zeros([3,3])
    rot_mat_y[0,0], rot_mat_y[0,2] = np.cos(np.radians(-theta_c)), -np.sin(np.radians(-theta_c))
    rot_mat_y[1,1] = 1
    rot_mat_y[2,0], rot_mat_y[2,2] = np.sin(np.radians(-theta_c)), np.cos(np.radians(-theta_c))

    ## rotate coordiantes
    rot_coords_couch = np.round(np.dot(rot_mat_y, rot_coords_gantry), decimals=precision)
    
    if single_point: return rot_coords_couch[0,:][0], rot_coords_couch[1,:][0], rot_coords_couch[2,:][0]
    
    return rot_coords_couch[0,:], rot_coords_couch[1,:], rot_coords_couch[2,:]

def expand_contour_3d(image, fraction):
   
    struct_size = int(2 * fraction + 1)
    struct_element = np.ones((struct_size, struct_size, struct_size), dtype=np.bool)
    
    # Perform dilation
    dilated_image = binary_dilation(image, structure=struct_element)
    
    return dilated_image.astype(int)

def create_histogram(data, fig_name, bins = None, ylabel=None, xlable=None, title=None, fig_dir= None):
    
    if fig_dir is None:
        fig_dir = os.path.join('temp','data','figures')
        if not os.path.exists(fig_dir): os.makedirs(fig_dir)
        
     # Calculate bins if not provided
    if bins is None:
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        bin_width = 2 * iqr * len(data) ** (-1/3)
        if bin_width != 0:
            bins = round((data.max() - data.min()) / bin_width)
        
    # Calculate statistics
    mean_val = data.mean()
    median_val = data.median()
    std_dev = data.std()
    min_val = data.min()
    max_val = data.max()
    
    # Format statistics for display
    stats_text = (f'$\mu = {mean_val:.2f}$\n'
                  f'$\sigma = {std_dev:.2f}$\n'
                  f'$\min = {min_val:.2f}$\n'
                  f'$\mathrm{{median}} = {median_val:.2f}$\n'
                  f'$\max = {max_val:.2f}$')
    
    plt.figure(facecolor='white')
    if bins is not None:
        sns.histplot(data, bins=bins, kde=False, color='gray')
    else:
        sns.histplot(data, kde=False, color='gray')
    plt.xlabel(xlable)
    if title is not None: plt.title(title)

    # Annotation box with the statistics
    plt.annotate(stats_text, xy=(1.05, 0.5), xycoords='axes fraction', 
                 fontsize=10, bbox=dict(boxstyle="round, pad=0.3", edgecolor='gray', facecolor='whitesmoke'))

    # Saving the figure with a tight layout
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.png'), dpi=300)
    
def str_to_bool(s):
    if s.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif s.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise ValueError(f"Cannot convert {s} to a boolean.")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [atoi(c) for c in re.split(r'(\d+)', text)]
             
def calculate_dsc(volume1, volume2):
        intersection = np.sum(volume1 & volume2)
        volume1_sum = np.sum(volume1)
        volume2_sum = np.sum(volume2)
        return (2. * intersection) / (volume1_sum + volume2_sum)                          
                                