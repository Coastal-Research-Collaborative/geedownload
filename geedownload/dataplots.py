
import os
import numpy as np
# from PIL import Image
# import cv2
import rasterio

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch





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




def visualize_normalized_image(normalized_image: np.ndarray) -> np.ndarray:
    """
    Stretches a standardized image to a viewable 0-1 range for visualization.
    
    This is for display purposes only and should NOT be used for training.
    """
    # Find the min and max values of the normalized image
    min_vals = np.min(normalized_image, axis=(0, 1))
    max_vals = np.max(normalized_image, axis=(0, 1))
    
    # Apply a Min-Max scaling to the normalized data
    stretched_image = (normalized_image - min_vals) / (max_vals - min_vals)
    
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


def plot_satellite_bands(file_list):
    """
    Plots satellite bands from a mixed list of multi-band and single-band TIFFs.

    Parameters:
    - file_list (list): A list of file paths. Assumes one multi-band file
                        and several single-band files, identifiable by name.
    """
    # Separate the files based on their names
    multiband_file = None
    singleband_files = []
    
    for f in file_list:
        if '.tif' in f and 'swir' not in f:
            # Assumes the multi-band file is the one without "swir" in the name
            multiband_file = f
        elif 'swir' in f:
            # Assumes single-band files have "swir" in their name
            singleband_files.append(f)
            
    # Raise an error if the multi-band file isn't found
    if not multiband_file:
        raise FileNotFoundError("Could not find the multi-band TIFF file (e.g., '...tif' without 'swir').")
    
    # Sort the single-band files to ensure consistent plotting order
    singleband_files.sort()
    
    # Determine the number of subplots
    num_subplots = 1 + len(singleband_files)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
    
    # Handle single subplot case
    if num_subplots == 1:
        axes = [axes]
        
    # Plot the multi-band TIFF (RGB and NIR)
    with rasterio.open(multiband_file) as src:
        # Read the first 4 bands for RGB and NIR
        tiff_array = src.read([1, 2, 3, 4])
        tiff_array = np.transpose(tiff_array, (1, 2, 0))
    
    # Plot RGB
    axes[0].imshow(rescale_image_intensity(tiff_array[:, :, [0, 1, 2]]))
    axes[0].set_title('RGB Image')

    # Plot NIR
    # Note: If your .tif file has RGB as bands 1,2,3 and NIR as band 4, this is correct.
    axes[1].imshow(rescale_image_intensity(tiff_array[:, :, 3]))
    axes[1].set_title('NIR Band')
    
    # Plot the single-band SWIR TIFFs
    for i, swir_file in enumerate(singleband_files):
        with rasterio.open(swir_file) as src:
            swir_array = src.read(1)
            
            # Extract band name from filename for title (e.g., "swir1" or "swir2")
            band_name = os.path.basename(swir_file).split('.')[1].upper()
            
            axes[i + 2].imshow(rescale_image_intensity(swir_array))
            axes[i + 2].set_title(f'{band_name} Band')
    
    plt.tight_layout()
    plt.show()


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