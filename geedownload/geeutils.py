"""
functions to download landsat and sentinel imagery from google earth engine

Joel Nicolow, Coastal Research Collaborative, November 2024
"""


import os
from glob import glob
import ee
import geojson
import requests
import zipfile 
import json
import numpy as np
import re
import tempfile
from pathlib import Path
import math
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds


from geedownload import tiffutils # used for cleaning up downloaded imagery files



def authenticate_and_initialize():
    """
    authenticate and initialize google earth engine
    """

    try: 
        ee.Initialize() # if there has already been an 
    except:
        # could attempt to refresh token via https://stackoverflow.com/questions/53472429/how-to-get-a-gcp-bearer-token-programmatically-with-python

        ee.Authenticate() # this will ask for a user input (if in vscode the input box will be at the top not inline)
        ee.Initialize()


def channel_name_to_band(channel_name, satname, reverse=False):
    """
    reverse goies from B1 etc to RGB etc
    """
    sat_dict = {
        # Landsat 5 (L5)
        'L5': {
            'B': 'B1',          # Blue
            'G': 'B2',          # Green
            'R': 'B3',          # Red
            'NIR': 'B4', # Near Infrared
            'SWIR1': 'B5',      # SWIR
            'TIR': 'B6', # Thermal infrared
            'SWIR2': 'B7',      # SWIR
            # 'PAN': 'B8', # this is used for panchromatic sharpening NOTE L5 does not have this band
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Landsat 7 (L7)
        'L7': {
            'B': 'B1',          # Blue
            'G': 'B2',          # Green
            'R': 'B3',          # Red
            'NIR': 'B4', # Near Infrared
            'SWIR1': 'B5',      # SWIR
            'TIR': 'B6_VCID_1', # Thermal infrared
            'SWIR2': 'B7',      # SWIR
            'PAN': 'B8', # this is used for panchromatic sharpening
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Landsat 8 (L8)
        'L8': {
            'B': 'B2',          # Blue
            'G': 'B3',          # Green
            'R': 'B4',          # Red
            'NIR': 'B5', # Near Infrared
            'SWIR1': 'B6',      # SWIR
            'SWIR2': 'B7',      # SWIR
            'PAN': 'B8', # this is used for panchromatic sharpening
            'TIR': 'B10', # themal infrared 
            'TIR2': 'B11',
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Landsat 9 (L9)
        'L9': {
            'B': 'B2',          # Blue
            'G': 'B3',          # Green
            'R': 'B4',          # Red
            'NIR': 'B5', # Near Infrared
            'SWIR1': 'B6',      # SWIR
            'SWIR2': 'B7',      # SWIR
            'PAN': 'B8', # this is used for panchromatic sharpening
            'TIR': 'B10', # themal infrared 
            'TIR2': 'B11',
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Sentinel-2 (S2)
        'S2': {
            'B': 'B2',          # Blue
            'G': 'B3',          # Green
            'R': 'B4',          # Red
            'NIR': 'B8', # Near Infrared
            'SWIR1': 'B11',     # SWIR1
            'SWIR2': 'B12',     # SWIR2
            'UDM':'QA60' # SCL is not really avalable for much imager
            # 'UDM': 'SCL' # this is the correctUDM band QA10 isnt really anything NOTE if SCL isnt abaible use QA60
            # 'UDM': 'QA10' # past work says this is basically the udm but it says not available'S2Cloudless' # Cloud Mask (using S2Cloudless algorithm)
        }
    }

    # Check if the satellite and channel are valid, and return the corresponding band
    if satname not in sat_dict:
        raise ValueError(f"Invalid satellite name '{satname}'")

    # Handle reverse lookup
    if reverse:
        # Flip the dictionary for the given satellite
        inverted_dict = {v: k for k, v in sat_dict[satname].items()}
        if channel_name in inverted_dict:
            return inverted_dict[channel_name]
        else:
            raise ValueError(f"Invalid band name '{channel_name}' for satellite '{satname}'")
    else:
        # Normal lookup
        if channel_name in sat_dict[satname]:
            return sat_dict[satname][channel_name]
        else:
            raise ValueError(f"Invalid channel name '{channel_name}' for satellite '{satname}'\n{sat_dict[satname]}")
        

#### handling too large AOI requests ####
def parse_request_size_from_error(error_str: str):
    """
    Extract actual and max bytes from GEE error message like:
    'Total request size (290499705 bytes) must be less than or equal to 50331648 bytes.'
    Returns (actual_bytes, max_bytes) or (None, None)
    """
    match = re.search(r'Total request size \((\d+) bytes\) must be less than or equal to (\d+) bytes', error_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def download_large_AOI_in_seperate_tiles(sitename:str, satname:str, bands:dict, aoi, image, image_id, size_error_str:str):

    if satname == 'S2':
        scale = 10
    elif satname in ('L5', 'L7', 'L8', 'L9'):
        scale = 30
    else:
        scale = 30

    # --- Calculate how many slices we need -----------------------------------
    if size_error_str:
        actual_bytes, max_bytes = parse_request_size_from_error(size_error_str)
        if actual_bytes and max_bytes:
            # e.g. 290MB / 50MB = 5.8 → need 6 slices to guarantee each is under limit
            n_slices = math.ceil(actual_bytes / max_bytes)
            print(f"  Image is {actual_bytes/1e6:.1f} MB, limit is {max_bytes/1e6:.1f} MB → need {n_slices} slices")
        else:
            n_slices = 4  # fallback
    else:
        n_slices = 4

    # --- Get AOI bounds -------------------------------------------------------
    coords = aoi.bounds().getInfo()['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Approximate width and height in metres
    lat_mid = (min_lat + max_lat) / 2
    width_m  = (max_lon - min_lon) * 111_320 * math.cos(math.radians(lat_mid))
    height_m = (max_lat - min_lat) * 111_320

    # --- Slice orthogonal to the longest side --------------------------------
    if width_m >= height_m:
        # Cut vertically (slice along longitude)
        # print(f"  Slicing vertically (width {width_m/1000:.1f} km > height {height_m/1000:.1f} km) into {n_slices} strips")
        lon_edges = np.linspace(min_lon, max_lon, n_slices + 1)
        slices = [
            (lon_edges[i], min_lat, lon_edges[i+1], max_lat)
            for i in range(n_slices)
        ]
    else:
        # Cut horizontally (slice along latitude)
        # print(f"  Slicing horizontally (height {height_m/1000:.1f} km > width {width_m/1000:.1f} km) into {n_slices} strips")
        lat_edges = np.linspace(min_lat, max_lat, n_slices + 1)
        slices = [
            (min_lon, lat_edges[i], max_lon, lat_edges[i+1])
            for i in range(n_slices)
        ]

    # --- Download each slice as a normal image --------------------------------

    for i, (s_min_lon, s_min_lat, s_max_lon, s_max_lat) in enumerate(slices):
        tile_region = ee.Geometry.Rectangle([s_min_lon, s_min_lat, s_max_lon, s_max_lat])
        image_id_tile = f'{tiffutils.get_timestamp(image_id, convert_format=True)}_tile_{i}'

        # try:
        url = image.getDownloadURL({
            'scale': scale,
            'region': tile_region.getInfo(),
            'bands': bands,
        })
        download_single_image(
            sitename=sitename,
            satname=satname,
            download_url=url,
            image_id=image_id,
            tile_number=i
        )
       

    # --- Mosaic combined tile tifs into one final image ----------------------
    sat_dir = os.path.join('data', 'sat_images', sitename, satname)
    timestamp_str = tiffutils.get_timestamp(image_id, convert_format=True)
    
    combined_tile_paths = sorted(glob(os.path.join(sat_dir, f'{satname}_*{timestamp_str}*_tile_*.tif')))
    combined_tile_paths = [p for p in combined_tile_paths if not any(
        p.endswith(f'.{band}.tif') for band in ['R', 'G', 'B', 'NIR', 'SWIR1', 'SWIR2', 'TIR', 'PAN', 'UDM']
    )]

    if not combined_tile_paths:
        raise RuntimeError(f"No combined tile tifs found for {timestamp_str}")

    # print(f"  Mosaicking {len(combined_tile_paths)} tiles: {[os.path.basename(p) for p in combined_tile_paths]}")

    final_path = os.path.join(sat_dir, f'{satname}_{timestamp_str}.tif')

    datasets = [rasterio.open(p) for p in combined_tile_paths]
    mosaic, transform = merge(datasets)

    # Copy meta + band descriptions from first tile
    meta = datasets[0].meta.copy()
    descriptions = datasets[0].descriptions  # tuple of band names e.g. ('R', 'G', 'B', ...)
    for ds in datasets:
        ds.close()

    meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': transform,
        'compress': 'lzw',
    })

    with rasterio.open(final_path, 'w', **meta) as dest:
        dest.write(mosaic)
        # Restore band descriptions so downstream code still sees R, G, B etc.
        for i, desc in enumerate(descriptions, start=1):
            dest.update_tags(i, name=desc)

    # print(f"  ✓ Mosaic saved → {final_path}")

    # Clean up tile tifs
    for p in combined_tile_paths:
        try:
            os.remove(p)
        except PermissionError:
            print(f"  Could not delete {os.path.basename(p)}")

    return final_path


def download_single_image(sitename:str, satname:str, download_url, image_id=None, alternate_save_path=None, tile_number:int=None):
    try:
        response = requests.get(download_url)
    except Exception as e:
        print('what is going on? requests.get exception')
        print(e)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # create download folder
        if alternate_save_path is None:
            download_folder_satname = os.path.join('data', 'sat_images', sitename, satname) # sitename dir was already mad
        else:
            download_folder_satname = alternate_save_path
        os.makedirs(download_folder_satname, exist_ok=True) # make sure the download folder exists before saving the file

        # change zip filename to include the satname at the beginning and avoid nested folders
        # image_id_fn = image_id.split("/")[-1]
        image_id_fn = tiffutils.get_timestamp(image_id, convert_format=False)
        # print(image_id_fn)


        zip_filename = os.path.join(download_folder_satname, f'{image_id_fn}_image.zip')
        
        with open(zip_filename, 'wb') as f:
            f.write(response.content)
        # print(f"File downloaded successfully as {zip_filename}")

        # Unzip the file into the download folder
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(download_folder_satname)  # Extract directly into the download folder
        # print(f"File unzipped successfully into {download_folder_satname}")

        # prepend satelite name to file names and replace channel with the actual channel
        this_image_component_fns = [] # these are each of the band names for the 
        
        # go through each of the bands
        tiff_fns = glob(os.path.join(download_folder_satname, f'*{image_id_fn}*'))
        for file_path in tiff_fns:
            if file_path.endswith('.zip'): continue # This is the zip file we took them out of
            if 'tile' in file_path: continue # this means its already been processed (the only possibility of this is if there are multiple tiles for same image timestamp)
            short_fn = os.path.basename(file_path)
            timestamp_str = tiffutils.get_timestamp(image_id, convert_format=True)
            # print(short_fn)
            period_split = short_fn.split('.')
            band = period_split[1] # last one is file extention
            # short_fn_no_band = period_split[0] # removes extention and band but has the defaultfilename version
            short_fn_no_band = timestamp_str
            if not tile_number is None:
                short_fn_no_band = f'{short_fn_no_band}_tile_{tile_number}'
            # print(band)
            try:
                short_fn = f'{short_fn_no_band}.{channel_name_to_band(channel_name=band, satname=satname, reverse=True)}'
            except ValueError:
                # For some satellites it may just 
                short_fn = f'{short_fn_no_band}.{band}' # names are already set to have the correct band name so no need reverse it
            # print(short_fn)

            new_filename = os.path.join(os.path.dirname(file_path), f"{satname}_{short_fn}.tif")

            if not file_path == new_filename: 
                if os.path.exists(new_filename):
                    # this mostlikely means this data was already downloaded
                    os.remove(new_filename) # NOTE it will now get overridden
                os.rename(file_path, new_filename) # NOTE done by resampling for landsat
            
            this_image_component_fns.append(new_filename) # save band file to list so fns can all be combined


        os.remove(zip_filename) # remove zip file

        imagery_downloaded = True # if any imagery is downloaded
        # print(this_image_component_fns)
        tiffutils.combine_tiffs(tiff_files=this_image_component_fns) # for each image combine band tiffs into one tiff file
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    
    return imagery_downloaded
    

def retrieve_imagery(sitename:str, start_date:str, end_date:str, data_dir=None, polygon=None, satnames:list=['L4', 'L5', 'L7', 'L8', 'L9', 'S2'], proccess_downloads:bool=True, specific_band_requests:dict=None, max_cloud_percent:int=20):
    """
    Download imagery for a given site (if no polygon loads sitename file)

    :param sitename: str the name of the site (used for where the images are downloaded)
    :param start_date: str "YYY-MM-DD" 
    :param end_date: str "YYY-MM-DD" 
    :param data_dir: str directory where sat_images/<sitename> is held. NOTE do not include sat_images or sitename in data_dir
    :param polygon: 2d list [longitude1, latitude1], [longitude2, latitude2], [longitude3, latitude3], [longitude4, latitude4]] NOTE does not need to be a rectangle
    :param satnames: list of strs the names of the satellites that we want to download imagery from
    :param proccess_downloads: bool if True then run tiffutils.clean_up_gee_downloads
    :param specific_band_requests: dict with satname and then what bands are requested if not None then this overwrites satnames
    :param max_cloud_percent: int (or float) max percent of the image that can be covered by clouds
    NOTE the combine bands function in tiffutils will only combine RGB NIR PAN and UDM all others will be left as their own bands
    """

    imagery_downloaded=False

    authenticate_and_initialize() # authenticate and initialize gee

    if data_dir is  None:
        download_folder = os.path.join('data', 'sat_images', sitename)
    else:
        download_folder = os.path.join(data_dir, 'sat_images', sitename)
    if not os.path.exists(download_folder): os.makedirs(download_folder)

    tiffutils.clean_up_gee_downloads(download_folder) # NOTE if some imagery was download prior but clean up wasn't run or the download was stopped early this will clean up misalenious files


    if polygon is None:
        # load from siteinfo
        # NOTE depending on the use case this structure may not be set up
        polygon_path = os.path.join('siteinfo', sitename, f'{sitename}_polygon.geojson')
        if not os.path.exists(polygon_path):
            polygon_path = os.path.join(data_dir, 'siteinfo', sitename, f'{sitename}_polygon.geojson')
        if not os.path.exists(polygon_path):
            print(polygon_path)
            raise('There is no polygon geojsonfiles in siteinfo/<sitename>/<sitename>_polygon.geojson or data/siteinfo/<sitename>/<sitename>_polygon.geojson')
        with open(polygon_path, 'r') as file: geojson_data = geojson.load(file)
        coords = geojson_data["features"][0]["geometry"]['coordinates'][0]
        polygon = [[coord[0], coord[1]] for coord in coords]  # Keep only lat, lon

    aoi = ee.Geometry.Polygon([polygon])

        

    # qa_band_Landsat = 'QA_PIXEL'
    # qa_band_S2 = 'QA60'
    # NOTE Default not actually ysed really
    sat_dict = {
        'L5': {'start_year': None, 'end_year': None, 'collection': 'LANDSAT/LT05/C02/T1_TOA'},
        'L7': {'start_year': None, 'end_year': 2022, 'collection': 'LANDSAT/LE07/C02/T1_TOA'},
        'L8': {'start_year': None, 'end_year': None, 'collection': 'LANDSAT/LC08/C02/T1_TOA'},
        'L9': {'start_year': 2022, 'end_year': None, 'collection': 'LANDSAT/LC09/C02/T1_TOA'},
        'S2': {'start_year': None, 'end_year': None, 'collection': 'COPERNICUS/S2_HARMONIZED'}
    }


    if not specific_band_requests is None:
        satnames = list(specific_band_requests.keys())
    for satname in satnames:
        if satname in sat_dict:
            sat_info = sat_dict[satname]
            if not specific_band_requests is None:
                # download the specifically requested bands
                bands = []
                for band in specific_band_requests[satname]:
                    bands.append(channel_name_to_band(band, satname))

            else:
                bands = [
                    channel_name_to_band('R', satname), 
                    channel_name_to_band('G', satname), 
                    channel_name_to_band('B', satname), 
                    channel_name_to_band('NIR', satname),
                    channel_name_to_band('UDM', satname) # NOTE dont want this for everytyhing because sentinel has wrong shape
                ]
                if not 'S' in satname and not satname == 'L5':
                    # landsat 5 doesnt have panchromatic band
                    # NOTE for sentinel the udm is like 8.99 m resolution while the rest is 10 m so explicitly ask for it in 10 m
                    # bands.append(channel_name_to_band('UDM', satname)) # landsat the udm should be in the right resolution naturally
                    bands.append(channel_name_to_band('PAN', satname)) # only landsat imagery has pan chromatic band
            
            # print(f'These are the bands for {satname}----------------------------------------')
            # print(bands)
                
            cloud_cover_term = 'CLOUD_COVER'
            if satname == 'S2': cloud_cover_term = 'CLOUDY_PIXEL_PERCENTAGE'
            collection = (ee.ImageCollection(sat_info['collection'])
                          .filterDate(start_date, end_date)
                          .filterBounds(aoi)
                          .filterMetadata(cloud_cover_term, 'less_than', max_cloud_percent)
                        )
            
            # Check if the collection is not empty
            try:
                n_images = collection.size().getInfo()
            except ee.ee_exception.EEException as e:
                n_images = 0 # if n_images = 0 (it will print out that this is because there are no images available)
            if n_images > 0:
                for image in collection.getInfo()['features']:   
                    # Get the ID of the image to download. This is each image not each band 
                    image_id = image['id'] # something like this: COPERNICUS/S2_HARMONIZED/20250712T153559_20250712T153728_T19TCG 
                    # print(f"Processing image: {image_id}")

                    image = ee.Image(image_id)
                    # print(image.bandNames().getInfo())


                    # scale = image.select(channel_name_to_band('R', satname)).projection().nominalScale().getInfo()
                    # # print(f'scale of red: {scale}')
                    # # print('----------------------------------------------------------------')
                    # if not 'S' in satname and not satname == 'L5':
                    #     scale = image.select(channel_name_to_band('PAN', satname)).projection().nominalScale().getInfo()
                    #     # print(f'scale of pancromatic: {scale}')
                    # elif satname == 'L5':
                    #     NOTE no panchromatic band for L5 so cant upsample resolution
                    #     print('no panchromatic band for L5 so cant upsample resolution')
                    # else:
                    #     # NOTE scale udm band for sentinal imagery cuz its 8.99 m instead of 10 m resolution
                    #     udm_band = channel_name_to_band('UDM', satname)
                    #     # Resample the UDM band to match the 10m resolution of the other bands
                    #     udm_resampled = (image.select(udm_band)
                    #                         .resample()  # Use 'bilinear' for continuous data, 'nearest' for categorical
                    #                         .reproject(crs=image.select(bands[0]).projection(), scale=10))
                    #     image = image.addBands(udm_resampled) # Add the resampled UDM band back to the image
                    #     bands.append(udm_band) # Add the UDM band back to the list of bands to export

                    # Prepare download URL
                    try:
                        if satname == 'S2':
                            scale, dtype_bytes = 10, 2   # uint16
                        elif satname in ('L5', 'L7'):
                            scale, dtype_bytes = 30, 2
                        elif satname in ('L8', 'L9'):
                            scale, dtype_bytes = 30, 4   # float32 — this was the underestimate bug
                        else:
                            scale, dtype_bytes = 30, 4
                        download_url = image.getDownloadURL({
                            'scale': scale,
                            'region': aoi.getInfo(),
                            'bands': bands
                        })
                    except Exception as e:
                        print('download url image.getDownloadURL issue. it it mentions size, reduce tile size')
                        print(e)
                        if 'Total request size (' in str(e) and '50331648' in str(e):
                            # this means the AOI is too big so it needs to be broken up into multiple AOIs
                            print('downloading scene in seperate tiles and then combining them to original AOI with download_large_AOI_in_seperate_tiles()')
                            download_large_AOI_in_seperate_tiles(sitename=sitename, satname=satname, bands=bands, aoi=aoi, image=image, image_id=image_id, size_error_str=str(e))
                            continue
                        else:
                            raise()

                    # print(f'Downloading these bands {bands}')
                    # print(f"Download URL: {download_url}")

                    # if 'S' in satname:
                    #     # NOTE udm band needs to be removed from bands each itteration because it is added above resampled as udm_resampled
                    #     bands.remove(udm_band)

                    download_single_image(sitename=sitename,
                                         satname=satname,
                                         download_url=download_url,
                                         image_id=image_id)
            else:
                print(f"No images found for {satname} in the given date range and polygon.")

    if proccess_downloads:
        tiffutils.clean_up_gee_downloads(download_folder)

    if imagery_downloaded:
        return True
    return False


def create_polygon_geojson(sitename:str, coords:list, data_dir:str='data', overwrite:bool=False):
    """
    Given a list of lat long coordinates this creates a polygon function used in the imagery download process
    """

    save_dir = os.path.join(data_dir, 'siteinfo', sitename)
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"{sitename}_polygon.geojson")

    if not overwrite and os.path.exists(save_path):
        print(f'Polygon geojson already exists at {save_path} and overwrite is set to False so not overwriting')
        return
    

    if coords[0] != coords[-1]:
        coords.append(coords[0])  # Close the polygon by repeating the first coordinate


    geojson_data = {
        "type": "FeatureCollection",
        "name": f"{sitename}_polygon",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "Name": "Polygon 1",
                    "description": None,
                    "timestamp": None,
                    "begin": None,
                    "end": None,
                    "altitudeMode": None,
                    "tessellate": -1,
                    "extrude": 0,
                    "visibility": -1,
                    "drawOrder": None,
                    "icon": None
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
        ]
    }

    
    with open(save_path, 'w') as geojson_file:
        json.dump(geojson_data, geojson_file, indent=4)

