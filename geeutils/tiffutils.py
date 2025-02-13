import os
from glob import glob
import gc
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal



def load_tiff_image(fn, plot=False, plot_scale=1):

    with rasterio.open(fn) as src:
        # Read the data
        image = src.read()  # This loads all bands as a numpy array
        # profile = src.profile  # Metadata about the file
        # Display some metadata
        # print(f"CRS: {src.crs}")
        # print(f"Bounds: {src.bounds}")
        print(image.shape)
        image_transposed = np.transpose(image, (1, 2, 0)) # get image in (h, w, channel)
        # print(f"Image shape: {image_transposed.shape}")

        if plot:
            min_val_rgb = image_transposed[:,:,:3].min()
            max_val_rgb = image_transposed[:,:,:3].max()
            min_val_nir = image_transposed[:,:,3].min()
            max_val_nir = image_transposed[:,:,3].max()
            min_val_udm = image_transposed[:,:,4].min()
            max_val_udm = image_transposed[:,:,4].max()

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 20))
            axes[0].imshow(image_transposed[:,:,[0,1,2]]/plot_scale)
            axes[0].set_title(f'{os.path.basename(fn)}, {image_transposed.shape}\nRGB min: {min_val_rgb}, max: {max_val_rgb}')
            axes[1].imshow(image_transposed[:,:,3]/plot_scale) # near infrared
            axes[1].set_title(f'NIR min: {min_val_nir}, max: {max_val_nir}')
            axes[2].imshow(image_transposed[:,:,4])
            axes[2].set_title(f'UDM min: {min_val_udm}, max: {max_val_udm}')
            
            fig.show()
            plt.show()

            # for channel in range(image_transposed.shape[2]):
            #     plt.imshow(image_transposed[:,:,channel]/plot_scale)
            #     plt.show()

        return(image_transposed)


def scale_band(image_band, satname:str=None):
    """
    if no sat name scales images between 0 and 1
    else: scales specific to a value for the given sat (then clips)
    """
    if satname is None:
        # Just min/max scale
        image_band_temp = np.nan_to_num(image_band, nan=0, posinf=0, neginf=0)
        min_val = np.min(image_band_temp)
        max_val = np.max(image_band_temp)
        image_band = np.nan_to_num(image_band, nan=0, posinf=max_val, neginf=min_val)

        if max_val != min_val:
            image_band = (image_band - min_val) / (max_val - min_val)
        else:
            image_band = np.zeros_like(image_band)  # return all zeros if constant
    else:
        # Do a scaling specific to different satellites
        if satname == 'S2':
            # image_band = image_band / 10_000 # this is what the imagery is natively scaled to 
            # image_band = np.clip(image_band, 0, 1) # make sure values are between 0 and 1 (they should be this is a "just in case")
            image_band = image_band # do nothing for now (maybe make it same as else)
        else:
            # NOTE for now this is the landsat satellites
            # landsat imagery is already between 0 and 1 but does have some inf values so just replaces these 
            image_band = np.nan_to_num(image_band, nan=0, posinf=1, neginf=0)
            # print('test doing nothing')
            
    
    return image_band


def create_rgb_image(red_image, green_image, blue_image, scale=False):
    red_image = np.squeeze(red_image)
    green_image = np.squeeze(green_image)
    blue_image = np.squeeze(blue_image)
    if scale:
        # Handle NaN, +inf, and -inf values by replacing them
        red_image_temp = np.nan_to_num(red_image, nan=0, posinf=0, neginf=0)
        green_image_temp = np.nan_to_num(green_image, nan=0, posinf=0, neginf=0)
        blue_image_temp = np.nan_to_num(blue_image, nan=0, posinf=0, neginf=0)
        red_image = np.nan_to_num(red_image, nan=0, posinf=np.max(red_image_temp), neginf=np.min(red_image_temp))
        green_image = np.nan_to_num(green_image, posinf=np.max(green_image_temp), neginf=np.min(green_image_temp))
        blue_image = np.nan_to_num(blue_image, nan=0, posinf=np.max(blue_image_temp), neginf=np.min(blue_image_temp))
        
        red_image = red_image / np.max(red_image)
        green_image = green_image / np.max(green_image)
        blue_image = blue_image / np.max(blue_image)
    rgb_image = np.stack((red_image, green_image, blue_image), axis=-1)
    return rgb_image


def combine_tiffs(tiff_files:list, output_path:str, satname=None, delete_original_files:bool=True, resample:bool=True, scale:bool=True):
    """
    This function gets the min and max pixel boundaries of a dataset

    Arguments:
    ----------
    tiff_files : list
        list of tiff filenames
    output_path : str
        path to where the output combined tiff will be saved
    satname : 
        None which mean it does generic minmax scaling but otherwise it does /10_000 for S2 (sentinel) and 
    delete_original_files : bool
        if True, then the single band files that were used to make the combined tiff will be delete
    resample : bool
        if True, the image bands will be sharpened to 
    scale : bool
        if True, convert -inf pixel values to 0
    
    Returns:
    --------
    tuple of four ints
    """

    # NOTE: sometimes for sentinel imagery it downloads duplicates of bands with the second one being empty
    print('what about this function?')
    tiff_files = remove_duplicate_band_files(fns = tiff_files, timestamp=None) # this returns the files that are good and removes the ones that are bad/duplicates (and deletes them)

    if resample:
        # seperate pan sharpened band
        pan_file = [file for file in tiff_files if file.endswith('.PAN.tif')]
        if len(pan_file)==0:
            resample=False # no resampling for sentinel this is a mistake
            del pan_file
            pan_dataset_dict = None
        else:
            # for landsat we have a pan band and use it to pansharpen the image
            pan_file = pan_file[0] # there will only ever be one (or zero of Sentinel 'S2')
            pan_dataset_dict = {'filename': pan_file, 'dataset': gdal.Open(pan_file)} # NOTE pan band is not saved as part of the output file
            tiff_files = [file for file in tiff_files if not file.endswith('.PAN.tif')]

    
    order = {'R': 0, 'G': 1, 'B': 2, 'NIR': 3, 'PAN': 4, 'UDM': 5}
    tiff_files = sorted(tiff_files, key=lambda x: order[x.split('.')[-2]]) # only for man images
    # Open all TIFF files as datasets

    datasets_dict_list = [{'filename': tiff, 'dataset': gdal.Open(tiff)} for tiff in tiff_files]

    # # Iterate through the list and print out the details
    # for item in datasets_dict_list:
    #     if item['dataset'] is not None:
    #         print(f"Filename: {item['filename']}, Dataset: {item['dataset']}")
    #     else:
    #         print(f"Filename: {item['filename']} - Dataset is None (null)")

    
    if resample:
        for i in range(len(datasets_dict_list)):
            # datasets does not include pan sharpened band
            resample_method = 'bilinear'
            pan_sharpen = True # Pan sharpen RGB bands

            if 'UDM' in datasets_dict_list[i]['filename']: 
                resample_method = 'nearest' # use nearest neighbor for UDM
                pan_sharpen = False # dont need pan sharpening for UDM
            if 'NIR' in datasets_dict_list[i]['filename']:
                # NIR band is sometimes pansharpened and sometimes not (not sure which to do)
                pan_sharpen = False
            temp_dataset = datasets_dict_list[i]['dataset']
            datasets_dict_list[i]['dataset'] = resample_in_memory(
                            input_dataset=datasets_dict_list[i]['dataset'],
                            target_dataset=pan_dataset_dict['dataset'],
                            double_res=True,  # Set to True if you want double resolution
                            resampling_method=resample_method,
                            apply_pansharpen=pan_sharpen
                            )             
        
    # Check that all datasets have the same CRS, bounds, and resolution
    datasets = [item.get('dataset') for item in datasets_dict_list] # extract just the datasets
    # del datasets_dict_list # we are not going to use the datasets_dict_list again
    # gc.collect() 
    crs_set = {ds.GetProjection() for ds in datasets}
    transform_set = {ds.GetGeoTransform() for ds in datasets}

    # Compare CRS
    if len(crs_set) > 1:
        print("CRS differences found at the following indices:")
        for i, ds in enumerate(datasets):
            if ds.GetProjection() != datasets[0].GetProjection():
                print(f"Dataset at index {i} has a different CRS: {ds.GetProjection()}")
                print(datasets_dict_list[i]['filename'])

    # Compare GeoTransform
    if len(transform_set) > 1:
        print("GeoTransform differences found at the following indices:")
        for i, ds in enumerate(datasets):
            if tuple(ds.GetGeoTransform()) != datasets[0].GetGeoTransform():
                print(f"Dataset at index {i} has a different GeoTransform: {ds.GetGeoTransform()}")
                print(datasets_dict_list[i]['filename'])

    resolution_set = {
        (abs(ds.GetGeoTransform()[1]), abs(ds.GetGeoTransform()[5])) for ds in datasets
    }

    
    create_combined_file = True
    if len(resolution_set) > 1:
        # NOTE the most common example of this is the udm band for sentinel imagery
        resolution_dict = {
            tiff_file: (abs(ds.GetGeoTransform()[1]), abs(ds.GetGeoTransform()[5]))
            for tiff_file, ds in zip(tiff_files, datasets)
        }

        print("TIFF files with their resolutions:")
        for file, res in resolution_dict.items():
            print(f"{file}: {res}")

        udm_index = next((i for i, item in enumerate(datasets_dict_list) if "UDM" in item["filename"]), None)
        
        if udm_index is not None:
            print(f"Original UDM has a different resolution: {datasets_dict_list[udm_index]['filename']}")
            
            # Remove the UDM dataset from processing
            datasets.pop(udm_index)

            # Generate new UDM in memory
            new_udm = generate_custom_udm(datasets)

            # Replace the old UDM dataset with the new UDM
            datasets_dict_list[udm_index]['dataset'] = None  # Remove old UDM
            datasets.append(new_udm)  # Append new UDM array to datasets list
        else:
            # this means sum band besides the udm has a different resolution so either skip this or raise and error
            print('deleteing these files because resolutions dont match for the different bands')
            # raise ValueError("Input TIFF files must have the same CRS, bounds, and resolution!")
            create_combined_file = False # the different bands are different resolitions so just deleteing everything
    if create_combined_file:
        # NOTE for now only do this if the resolutions are the same
        # create the output dataset based on the first dataset
        ref_dataset = datasets[0]
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(
            output_path,
            ref_dataset.RasterXSize,
            ref_dataset.RasterYSize,
            len(tiff_files),  # Number of layers
            ref_dataset.GetRasterBand(1).DataType,
        )
        output_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
        output_dataset.SetProjection(ref_dataset.GetProjection())

        # write each file (band) as a separate layer in the output dataset
        band_descriptions = ['Red', 'Green', 'Blue', 'NIR', 'UDM']
        for idx, ds in enumerate(datasets, start=1):
            band_data = ds.GetRasterBand(1).ReadAsArray()
            if scale:
                band_data = scale_band(band_data, satname=satname)
            output_band = output_dataset.GetRasterBand(idx)
            output_band.WriteArray(band_data)
            output_band.SetDescription(band_descriptions[idx-1]) # NOTE because enumerator starts at 1

        # close datasets to release resources -------------------------------------------------
        output_dataset.FlushCache()
        output_dataset = None  # close output file
       

    # this is done even if we cant combine the channels
    # close datasets to release resources -------------------------------------------------
    for ds in datasets:
        ds.FlushCache()
        ds = None  # close input files
        del ds  # Ensure reference is deleted

    del datasets
    gc.collect() # this is neccesary to avoid permission issue with deleting original file

    if delete_original_files:
        for tiff in tiff_files:
            try:
                os.remove(tiff)
            except(PermissionError):
                # This only happens for sentinel but it says permision denied
                print('permission denied stoopid')
        if resample and not pan_dataset_dict is None:
            del pan_dataset_dict
            gc.collect()
            if os.path.exists(pan_file): os.remove(pan_file)


def get_timestamp(fn, convert_format=False) -> str:
    """
    returns timestamp st in this format YYYYMMDD_HHmmSS unless fn says original format
    NOTE: takes the first of the timestamps for sentinel
    """
    if '/' in fn or '\\' in fn:
        # this means its a file path
        fn = os.path.basename(fn)
    
    first_split = fn.split('_')
    satname = first_split[0]
    # print(satname)
    if satname.startswith('L'):
        # Landsat L7_LE07_089083_20191114.B where 08 is the hour and 2019 is year etc.
        timestamp = first_split[2]
        date = first_split[-1].split('.')[0]
        timestamp_str = f'{timestamp}_{date}'            
    elif satname.startswith('S'):
        # Sentinel is in this format S2_20191101T000241_20191101T000243_T56HLH.B where the first time is start of aquisition and the second is end of aqwuizition in utc time
        timestamp_str = fn.split('_')[1] # using start of image aquisition timestamp

    if convert_format:
        convert_raw_timestamp(timestamp_str, satname)

    # print(date)
    # print(timestamp)
    return timestamp_str


def convert_raw_timestamp(timestamp_str, satname):
    if satname.startswith('S'):
        date, time = timestamp_str.split('T')
        timestamp_str = f'{date}_{time}'
    elif satname.startswith('L'):
        time, date = timestamp_str.split('_')
        timestamp_str = f'{date}_{time}'

    return timestamp_str


def get_image_bounds(dataset:gdal.Dataset) -> tuple:
    """
    This function gets the min and max pixel boundaries of a dataset

    Arguments:
    ----------
    dataset : gdal.Dataset
        The dataset which these boundaries are to be computed on
    
    Returns:
    --------
    tuple of four ints
    """
    geotransform = dataset.GetGeoTransform()
    x_min = geotransform[0]
    x_max = x_min + dataset.RasterXSize * geotransform[1]
    y_max = geotransform[3]
    y_min = y_max + dataset.RasterYSize * geotransform[5]
    return x_min, y_min, x_max, y_max


def resample_in_memory(input_dataset:gdal.Dataset, target_dataset:gdal.Dataset, double_res:bool=True, resampling_method:str='bilinear', apply_pansharpen:bool=True) -> gdal.Dataset:
    """
    Resample an input dataset to match the resolution and extent of a target dataset (usually a pan chromatic band) using GDAL Warp in-memory.

    Arguments:
    ----------
    input_dataset : gdal.Dataset
        The input dataset to be resampled.
    target_dataset : gdal.Dataset
        The target dataset to match extent and resolution.
    double_res : bool, optional
        If True, doubles the resolution (halves the pixel size).
    resampling_method : str, optional
        Resampling algorithm (e.g., 'bilinear', 'nearest').
    apply_pansharpen : bool, optional
        If True, target_dataset is assumed to be the panchromatic band and panchromatic sharpening is applied

    Returns:
    --------
    gdal.Dataset : The resampled dataset in memory.
    """
    # Get geotransform and resolution of the target dataset
    target_georef = np.array(target_dataset.GetGeoTransform())
    x_res = target_georef[1]
    y_res = target_georef[5]

    # Get bounds of the target dataset
    xmin, ymin, xmax, ymax = get_image_bounds(target_dataset)

    # Adjust resolution for double_res
    if double_res:
        input_georef = np.array(input_dataset.GetGeoTransform())

        downsample_x_res = input_georef[1] * 2  # Double pixel size (half resolution)
        downsample_y_res = abs(input_georef[5]) * 2
        downsample_options = gdal.WarpOptions(
            format='MEM',
            xRes=downsample_x_res,
            yRes=downsample_y_res,
            outputBounds=[xmin, ymin, xmax, ymax],
            resampleAlg=resampling_method,
            targetAlignedPixels=False
        )
        input_dataset = gdal.Warp('', input_dataset, options=downsample_options)

    

    # GDAL Warp options
    options = gdal.WarpOptions(
        format='MEM',  # Use 'MEM' driver for in-memory dataset
        xRes=x_res,
        yRes=abs(y_res),
        outputBounds=[xmin, ymin, xmax, ymax],
        resampleAlg=resampling_method,
        targetAlignedPixels=False
    )

    # perform resampling
    output_dataset = gdal.Warp('', input_dataset, options=options)

    if apply_pansharpen:
        resampled_array = output_dataset.ReadAsArray().astype(np.float32)
        pan_array = target_dataset.ReadAsArray().astype(np.float32)

        # ratio-based pansharpening
        # pan_array[np.isneginf(pan_array)] = 0 # NOTE instead (in line bellow) just ignoring these sections of the image (which are usually unusable data)
        pan_mean = np.mean(pan_array[~np.isneginf(pan_array)]) # sometimes there are iinf values if there are sections of images missing and this throws off the mean
        pan_ratio = (pan_array + 1e-6) / (pan_mean + 1e-6) # 1e-6 to avoid devide by 0
        pansharpened_array = resampled_array * pan_ratio

        # write pan-sharpened array to dataset
        driver = gdal.GetDriverByName('MEM')
        pansharpened_dataset = driver.Create(
            '',
            output_dataset.RasterXSize,
            output_dataset.RasterYSize,
            1,
            gdal.GDT_Float32
        )
        pansharpened_dataset.SetGeoTransform(output_dataset.GetGeoTransform())
        pansharpened_dataset.SetProjection(output_dataset.GetProjection())
        pansharpened_dataset.GetRasterBand(1).WriteArray(pansharpened_array)

        return pansharpened_dataset

    return output_dataset


def generate_custom_udm(datasets):
    """
    Generate a UDM mask where pixels that are zero across all bands are set to 1 in UDM.

    Args:
        datasets (list of gdal.Dataset): List of in-memory datasets for bands (excluding original UDM).

    Returns:
        np.ndarray: Generated UDM mask.
    """
    # Read all bands into memory as numpy arrays
    band_arrays = [ds.GetRasterBand(1).ReadAsArray() for ds in datasets if ds is not None]

    # Stack bands into a 3D array: shape (bands, height, width)
    band_stack = np.array(band_arrays)

    # Generate UDM: Mark pixels where ALL bands are 0 as 1, else 0
    udm_array = np.all(band_stack == 0, axis=0).astype(np.uint8)

    # Get metadata from first dataset
    ref_dataset = datasets[0]
    cols, rows = ref_dataset.RasterXSize, ref_dataset.RasterYSize
    geotransform = ref_dataset.GetGeoTransform()
    projection = ref_dataset.GetProjection()

    # Create an in-memory GDAL dataset
    driver = gdal.GetDriverByName("MEM")
    udm_dataset = driver.Create("", cols, rows, 1, gdal.GDT_Byte)  # 1-band UDM, Byte type

    # Set metadata
    udm_dataset.SetGeoTransform(geotransform)
    udm_dataset.SetProjection(projection)

    # Write UDM data to band
    udm_dataset.GetRasterBand(1).WriteArray(udm_array)

    return udm_dataset  # Now returns a proper GDAL dataset

def plot_datasets(datasets_dict_list, pan_dataset_dict=None):
    ncols = len(datasets_dict_list) 
    if not pan_dataset_dict is None: ncols += 1 # add a col for pansharpened band
    fig, axes = plt.subplots(nrows = 1, ncols=ncols, figsize=(15, 10))
    for i, dataset_dict in enumerate(datasets_dict_list):
        # Read the band data as a numpy array
        band = dataset_dict['dataset'].GetRasterBand(1)  # Get the first band
        band_data = band.ReadAsArray()  # Convert raster band to NumPy array
        band_name = dataset_dict['filename'].split('.')[-2]  # Extracting band type (e.g., R, G, B)
        # Plot the band
        axes[i].imshow(band_data, cmap='gray')
        # axes[i].colorbar(label='Pixel Value') # cant do this with axes
        axes[i].set_title(f"Band: {band_name} (Dataset {i+1})")
        # axes[i].set_xlabel("Column Index")
        # axes[i].set_ylabel("Row Index")

    # Optionally plot the panchromatic band
    if pan_dataset_dict:
        pan_dataset = pan_dataset_dict['dataset'].GetRasterBand(1)
        pan_data = pan_dataset.ReadAsArray()
        axes[i+1].imshow(pan_data, cmap='gray')
        # axes[i+1].colorbar(label='Pixel Value')
        axes[i+1].set_title("Panchromatic Band")
        # axes[i+1].set_xlabel("Column Index")
        # axes[i+1].set_ylabel("Row Index")
    plt.tight_layout()
    plt.show()


def remove_duplicate_band_files(fns, timestamp=None):
    """
    For the sentinel downloads sometimes there are to versions of each band downloaded so we want to delete one of them before combining the bands into one thing
    NOTE fns should all be in the same directory and be for the same timestamp
    For sentinel 2 imagery they download and extra copy of each band that is null if you try to open it (for now just deleting the second one is sufficient but at a later point maybe need to have a smarter function)
    """

    bands = set()
    fns_filtered = []

    # get bands
    for fn in fns:
        bands.add(os.path.basename(fn).split('.')[1])
    if not timestamp is None:
        for band in bands:
            print(band)
            band_fns = glob(os.path.join(os.path.dirname(fns[0]), f'*{timestamp}*.{band}.tif'))
            fns_filtered.append(band_fns[0])
            if len(band_fns) > 1:
                # This means there are duplicates for the same timestamp just delete one of them (but make sure this is consistant across bands)
                # delete all but first
                for band_fn in band_fns[1:]:
                    os.remove(band_fn)     
    else:
        # This is just for doing with multiple timesheets
        timestamps = {get_timestamp(fn) for fn in fns} # a set
        print(timestamps)
        for timestamp in timestamps:
            for band in bands:
                band_fns = glob(os.path.join(os.path.dirname(fns[0]), f'*{timestamp}*.{band}.tif'))
                fns_filtered.append(band_fns[0])
                if len(band_fns) > 1:
                    # delete all but first
                    for band_fn in band_fns[1:]:
                        os.remove(band_fn)     
                # for band_fn in band_fns:
                #     if 'QEK' in band_fn:
                #         # NOTE: just temparrary because these seem to be the broken files: data\sat_images\hawaiisharkscoveoahu\S2\S2_20191120T211919_20191120T211914_T04QEK.NIR.tif
                #         os.remove(band_fn)
                # if len(band_fns) > 1:
                #     # This means there are duplicates for the same timestamp just delete one of them (but make sure this is consistant across bands)
                #     for band_fn in band_fns:
                #         dataset = gdal.Open(band_fn)
                #         if dataset is None:
                #             os.remove(band_fn)
                #             print(f'Deleting: {band_fn}')
                #         else:
                #             dataset = None # this should be a way to close this
    return fns_filtered


def del_leftover_band_files(data_dir):
    """
    This function deletes left over single band files from the gee downloaded images

    sometimes there are permission errors with eleting in tiffutils.combine_tiffs() and in that case restart enve and run this
    """
    tiff_files = glob(os.path.join(data_dir, '*', '*.tif')) # the * for dir is the say name folder
    for tiff_fn in tiff_files:
        delete = False
        if '.R.tif' in tiff_fn:
            delete = True
        if '.G.tif' in tiff_fn:
            delete = True
        if '.B.tif' in tiff_fn:
            delete = True
        if '.NIR.tif' in tiff_fn:
            delete = True
        if '.PAN.tif' in tiff_fn:
            delete = True
        if '.UDM.tif' in tiff_fn:
            delete = True
        
        if delete:
            os.remove(tiff_fn)


def clean_up_gee_downloads(data_dir):

    for satname in os.listdir(data_dir):
        print(f'{satname}-------------------------')
        sat_data_dir = os.path.join(data_dir, satname)
        # if satname == 'S2': continue

        red_fns = glob(os.path.join(sat_data_dir, f'{satname}_*_*.R.tif'))
        if len(red_fns) == 0: continue # this means it was prolly already converted

        timestamps = [get_timestamp(fn) for fn in red_fns] 

        for timestamp in timestamps:

            glob_pattern = os.path.join(sat_data_dir, f'{satname}*{timestamp}*.*.tif') # this is general so it works for sentinel and landsat

            fns = glob(glob_pattern)

            # NOTE: There may be duplicates for the same timestamp (e.g. S2_20191105T211921_20191105T211919_T04QEJ.B, S2_20191105T211921_20191105T211919_T04QEK.B)
            # check for these duplicates and pick one and delete the others
            remove_duplicate_band_files(fns, timestamp)

            timestamp_str = convert_raw_timestamp(timestamp_str=timestamp, satname=satname) # using one timestamp format for all satelittes

            save_path = os.path.join(data_dir, satname, f'{satname}_{timestamp_str}.tif') # this gets rid of the LC08 or what ever other weird addition there is in the data
            resample = True
            if satname.startswith('S'): resample = False # just resample for landsat (not for sentinel images)
            combine_tiffs(fns, output_path=save_path, satname=satname, scale=True, resample=resample)
        
    del_leftover_band_files(data_dir)
