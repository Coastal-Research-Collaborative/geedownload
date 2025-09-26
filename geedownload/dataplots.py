
import os
import numpy as np
# from PIL import Image
# import cv2
import rasterio

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image






def add_mask_to_ax(ax, target, classes=None, title='Class Regions', add_legend=True):

    if target.shape[0] != 256 and target.ndim == 3 and target.shape[1] == 256 and target.shape[2] == 256:
        target = np.moveaxis(target, 0, -1)  # Moves first axis to the last, shape becomes (H, W, C)

    if len(target.shape) == 2 or target.shape[2] == 1:  # Binary mask (single channel)
        mask_img = target[:, :, 0]  # shape (H, W)
        im = ax.imshow(mask_img, cmap='viridis')
        if not title is None:
            ax.set_title(title)
        if add_legend:
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return
    
    
    cmap = plt.cm.get_cmap('tab20')  # Use a discrete color map suitable for categorical data
    # colors = [(253/255,231/255,37/255, 1), (68/255,2/255,86/255, 1)]

    # color_map = {
    #     'cloud': (235/255, 241/255, 244/255),  # white
    #     'sand': (194/255, 178/255, 128/255),  # sand color
    #     'otherland': (139/255, 69/255, 19/255),  # brown
    #     'water': (175/255, 219/255, 240/255)  # blue
    # } # NOTE temp for four class

    mask_img = np.zeros((*target[:, :, 0].shape, 3))
    legend_patches = []
    for i in range(target.shape[2]):
        class_region = target[:, :, i] > 0.5 # NOTE just incase there is a soft label some how

        color = cmap(i / target.shape[2])[:3]  # Normalize index and extract RGB
        # class_name = classes[i] if not classes is None else f'Class {i}'
        # color = color_map.get(class_name, (0, 0, 0))  
        # if len(classes) == 2:
        #     color = colors[i]
        mask_img[class_region == 1] = mcolors.to_rgb(color)  # Assign color to each class

        # Add a patch for each class for the legend
        if not classes is None and add_legend:
            legend_patches.append(Patch(color=color, label=classes[i])) # if no classes just plot it without labels

    # Show the class regions with specific colors in the second subplot
    ax.imshow(mask_img)
    if not title is None: ax.set_title(title)

    # Add the legend to the second subplot
    if add_legend:
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def visualize_normalized_image(normalized_image: np.ndarray, brightness: float = 0.2) -> np.ndarray:
    """
    Stretches a standardized image to a viewable 0-1 range for visualization with brightness adjustment.
    
    This is for display purposes only and should NOT be used for training.
    """
    # Find the min and max values of the normalized image
    min_vals = np.min(normalized_image, axis=(0, 1))
    max_vals = np.max(normalized_image, axis=(0, 1))
    
    # Apply a Min-Max scaling to the normalized data
    stretched_image = (normalized_image - min_vals) / (max_vals - min_vals)
    
    # Add the brightness term
    stretched_image = stretched_image + brightness
    
    # Clip values to ensure they are within the 0-1 range
    return np.clip(stretched_image, 0, 1)

def plot_tiff_img(tiff_fn, ax=None):
    with rasterio.open(tiff_fn) as dataset: tiff_array = dataset.read()
    tiff_array = np.transpose(tiff_array, (1, 2, 0)) # naturall first demension is channels

    if ax is None:
        fig, ax = plt.subplots(1, 2)

    # Display the tiff image
    ax[0].imshow(rescale_image_intensity(tiff_array[:,:, [0,1,2]]))
    ax[0].set_title('RGB Image')

    ax[1].imshow(rescale_image_intensity(tiff_array[:,:, 3]))
    ax[1].set_title('NIR Band')

    plt.show()


def plot_tiff_img_extra_bands(rgb_tiff_fn, extra_tiff_fns, extra_band_names):
    """
    Plots an RGB image and additional bands from separate single-band TIFF files.

    Parameters:
    - rgb_tiff_fn (str): The path to the TIFF file containing the RGB bands.
    - extra_tiff_fns (list of str): A list of paths to the single-band TIFF files.
    - extra_band_names (list of str): A list of names for the extra bands. Must
                                      have the same length as extra_tiff_fns.
    """
    # Validate input lists
    if len(extra_tiff_fns) != len(extra_band_names):
        raise ValueError("The number of extra TIFF files must match the number of band names.")

    # Determine the number of subplots
    num_subplots = 1 + len(extra_tiff_fns)

    # Create the figure and axes
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

    # Ensure axes is an array even for a single subplot
    if num_subplots == 1:
        axes = [axes]

    # Plot the RGB image on the first subplot
    with rasterio.open(rgb_tiff_fn) as dataset:
        rgb_array = dataset.read()
    rgb_array = np.transpose(rgb_array, (1, 2, 0))
    axes[0].imshow(rescale_image_intensity(rgb_array))
    axes[0].set_title('RGB Image')

    # Plot the additional bands in a loop
    for i, tiff_fn in enumerate(extra_tiff_fns):
        with rasterio.open(tiff_fn) as dataset:
            # Read the single band from the TIFF
            band_array = dataset.read(1)
            axes[i + 1].imshow(rescale_image_intensity(band_array))
            axes[i + 1].set_title(extra_band_names[i])

    plt.tight_layout()
    plt.show()


def apply_viridis_colormap(image_array):
    """
    Applies the viridis colormap to a single-band image array.
    """
    normalized_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    colormap = cm.get_cmap('viridis')
    viridis_array = colormap(normalized_array)
    return viridis_array[:, :, :3]

def plot_satellite_bands(file_list, crop_dim=None, save_dir=None):
    """
    Plots and optionally saves satellite bands to a specified directory.
    Plots single-band images with the viridis colormap.

    Parameters:
    - file_list (list): A list of file paths. Assumes one multi-band file
                        and several single-band files, identifiable by name.
    - crop_dim (int, optional): The side length for a square crop. If provided,
                                all images will be cropped to this dimension.
    - save_dir (str, optional): The directory to save the cropped and separated
                                bands as PNG files.
    """
    multiband_file = None
    singleband_files = []
    
    for f in file_list:
        if '.tif' in f and 'swir' not in f:
            multiband_file = f
        elif 'swir' in f:
            singleband_files.append(f)
            
    if not multiband_file:
        raise FileNotFoundError("Could not find the multi-band TIFF file (e.g., '...tif' without 'swir').")
    
    singleband_files.sort()
    
    with rasterio.open(multiband_file) as src:
        multiband_array_raw = src.read()
    
    if crop_dim:
        height, width = multiband_array_raw.shape[1], multiband_array_raw.shape[2]
        start_h = max(0, (height - crop_dim) // 2)
        start_w = max(0, (width - crop_dim) // 2)
        multiband_array = multiband_array_raw[:, start_h:start_h + crop_dim, start_w:start_w + crop_dim]
    else:
        multiband_array = multiband_array_raw

    swir_arrays = []
    swir_names = []
    for swir_file in singleband_files:
        with rasterio.open(swir_file) as src:
            swir_array_raw = src.read(1)
            
            if crop_dim:
                height, width = swir_array_raw.shape[0], swir_array_raw.shape[1]
                start_h = max(0, (height - crop_dim) // 2)
                start_w = max(0, (width - crop_dim) // 2)
                swir_array = swir_array_raw[start_h:start_h + crop_dim, start_w:start_w + crop_dim]
            else:
                swir_array = swir_array_raw
            
            swir_arrays.append(swir_array)
            band_name = os.path.basename(swir_file).split('.')[1].upper()
            swir_names.append(band_name)

    # Convert multiband array for plotting/saving
    multiband_array_transposed = np.transpose(multiband_array, (1, 2, 0))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(multiband_file).split('.')[0]

        # Save RGB composite
        rgb_composite_array = multiband_array_transposed[:, :, [0, 1, 2]]
        # Use your new function for a better visual result
        visualized_rgb = visualize_normalized_image(rgb_composite_array)
        Image.fromarray((visualized_rgb * 255).astype(np.uint8)).save(os.path.join(save_dir, f'{base_name}_rgb.png'))

        # Save individual RGB channels
        channel_names = ['blue', 'green', 'red']
        for i, name in enumerate(channel_names):
            # Transpose to (H, W, C) for visualization function
            single_channel_array_transposed = np.expand_dims(multiband_array[i, :, :], axis=-1)
            
            # Use your new function and apply viridis colormap
            visualized_channel = visualize_normalized_image(single_channel_array_transposed)
            viridis_channel = apply_viridis_colormap(visualized_channel[:, :, 0])
            
            # Convert to uint8 for saving
            Image.fromarray((viridis_channel * 255).astype(np.uint8)).save(os.path.join(save_dir, f'{base_name}_{name}.png'))
            

        # Save NIR band
        nir_array = multiband_array[3, :, :]
        viridis_nir = apply_viridis_colormap(nir_array)
        Image.fromarray((viridis_nir * 255).astype(np.uint8)).save(os.path.join(save_dir, f'{base_name}_nir.png'))
        
        for i, swir_array in enumerate(swir_arrays):
            # Save SWIR bands
            viridis_swir = apply_viridis_colormap(swir_array)
            Image.fromarray((viridis_swir * 255).astype(np.uint8)).save(os.path.join(save_dir, f'{base_name}_{swir_names[i].lower()}.png'))
            
    num_subplots = 2 + len(swir_arrays)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
    
    if num_subplots == 1:
        axes = [axes]
    
    axes[0].imshow(visualize_normalized_image(multiband_array_transposed[:, :, [0, 1, 2]]))
    axes[0].set_title('RGB Image')

    axes[1].imshow(multiband_array_transposed[:, :, 3], cmap='viridis')
    axes[1].set_title('NIR Band')
    
    for i, swir_array in enumerate(swir_arrays):
        axes[i + 2].imshow(swir_array, cmap='viridis')
        axes[i + 2].set_title(f'{swir_names[i]} Band')
    
    plt.tight_layout()
    plt.show()

# Example usage
# file_list = [
#     'data\\sat_images\\michigangreatbearlake\\S2\\S2_20240804_162839.swir1.tif',
#     'data\\sat_images\\michigangreatbearlake\\S2\\S2_20240804_162839.swir2.tif',
#     'data\\sat_images\\michigangreatbearlake\\S2\\S2_20240804_162839.tif'
# ]
# plot_satellite_bands(file_list, crop_dim=512, save_dir='media')

import skimage.exposure as exposure

def rescale_image_intensity(im, cloud_mask=None, prob_high=99.9):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """
    if cloud_mask is None:
        cloud_mask = np.zeros((im.shape[0], im.shape[1])).astype(bool)
    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high),
                                                      out_range = (0,1))  # YD
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj