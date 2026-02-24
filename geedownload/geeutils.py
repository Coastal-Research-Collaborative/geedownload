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
import math
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
import tempfile
from pathlib import Path

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
def estimate_tile_count(bbox, scale_m, n_bands=4, dtype_bytes=2):
    """
    Estimate how many tiles we need to stay under GEE's ~48MB limit.
    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # Approximate pixel counts
    lat_m = (max_lat - min_lat) * 111_320
    lon_m = (max_lon - min_lon) * 111_320 * math.cos(math.radians((min_lat + max_lat) / 2))
    n_pixels = (lat_m / scale_m) * (lon_m / scale_m)
    total_bytes = n_pixels * n_bands * dtype_bytes
    limit_bytes = 48 * 1024 * 1024  # 48 MB to be safe
    n_tiles = math.ceil(total_bytes / limit_bytes)
    # Round up to a perfect square grid
    grid_side = math.ceil(math.sqrt(n_tiles))
    return grid_side


def make_tile_grid(bbox, grid_side, overlap_deg=0.001):
    """
    Split bbox into a grid_side x grid_side grid of overlapping tiles.
    overlap_deg prevents seam artifacts at tile edges.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / grid_side
    lat_step = (max_lat - min_lat) / grid_side

    tiles = []
    for row in range(grid_side):
        for col in range(grid_side):
            t_min_lon = min_lon + col * lon_step - overlap_deg
            t_max_lon = min_lon + (col + 1) * lon_step + overlap_deg
            t_min_lat = min_lat + row * lat_step - overlap_deg
            t_max_lat = min_lat + (row + 1) * lat_step + overlap_deg

            # Clamp to original bbox
            tiles.append((
                max(t_min_lon, min_lon),
                max(t_min_lat, min_lat),
                min(t_max_lon, max_lon),
                min(t_max_lat, max_lat),
            ))
    return tiles


def mosaic_tiles(tile_paths: list, out_path: str):
    """
    Merge a list of GeoTIFF tile paths into a single output GeoTIFF.
    Uses rasterio.merge which handles overlapping regions by taking the
    first valid pixel (no blending seams).
    """
    datasets = [rasterio.open(p) for p in tile_paths]

    mosaic, transform = merge(datasets)

    # Copy metadata from first tile
    meta = datasets[0].meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "compress": "lzw",       # lossless compression
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    for ds in datasets:
        ds.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dest:
        dest.write(mosaic)


def download_large_AOI_in_seperate_tiles(sitename:str, satname:str, bands:dict, aoi, image, image_id):
    coords = aoi.bounds().getInfo()['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox = (min(lons), min(lats), max(lons), max(lats))
    print(coords)
    if satname == 'S2':
        scale, dtype_bytes = 10, 2   # uint16
    elif satname in ('L5', 'L7'):
        scale, dtype_bytes = 30, 2
    elif satname in ('L8', 'L9'):
        scale, dtype_bytes = 30, 4   # float32 — this was the underestimate bug
    else:
        scale, dtype_bytes = 30, 4
    grid_side = estimate_tile_count(
        bbox,
        scale_m=scale,
        n_bands=len(bands),
        dtype_bytes=dtype_bytes  # int16 for Sentinel, bump to 4 for float32
    )
    print(f"  Grid: {grid_side}×{grid_side} = {grid_side**2} tiles")\
    
    tiles = make_tile_grid(bbox, grid_side)
    print(tiles)

    temp_download_folder = os.path.join('data', 'sat_images', sitename, f'{satname}_tiled')
    os.makedirs(temp_download_folder, exist_ok=True)
    tile_paths = []

    for i, (min_lon, min_lat, max_lon, max_lat) in enumerate(tiles):
        tile_region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        # image_id_tile = f"{image_id}_tile_{i}"
        # tile_path = os.path.join(temp_download_folder, image_id_tile)
        for attempt in range(3):
            try:    
                try:
                    url = image.getDownloadURL({
                        'scale': scale,
                        'region': tile_region.getInfo(),
                        'bands': bands,
                    })
                    print(f"  URL generated OK: {url[:80]}...")
                except Exception as e:
                    print('❌ getDownloadURL failed — tile still too big?')
                    print(e)
                    break  # no point retrying if URL generation itself failed

                download_single_image(sitename=sitename, 
                                      satname=satname,
                                      download_url=url,
                                      image_id=image_id,
                                    #   alternate_save_path=temp_download_folder # should actually save to the regular place
                                      )


            except rasterio.errors.RasterioIOError as e:
                print(e)
                print(f"  ✗ Tile {i+1} attempt {attempt+1}: file not a valid GeoTIFF — likely GEE returned an error body. {e}")
                # if os.path.exists(tile_path):
                #     os.remove(tile_path)  # delete corrupt file so it doesn't get reused
            except Exception as e:
                print(e)
                print(f"  ✗ Tile {i+1} attempt {attempt+1}: {e}")
                # if os.path.exists(tile_path):
                #     os.remove(tile_path)


    # ✅ These are now OUTSIDE the for loop — mosaic after ALL tiles downloaded
    if not tile_paths:
        raise RuntimeError("All tiles failed to download — cannot mosaic.")

    print(f"  Mosaicking {len(tile_paths)} tiles...")
    mosaic_path = os.path.join(temp_download_folder, "mosaic.tif")
    mosaic_tiles(tile_paths, mosaic_path)

    for p in tile_paths:
        os.remove(p)

    print(f"  ✓ Mosaic saved to {mosaic_path}")
    return mosaic_path




def download_single_image(sitename:str, satname:str, download_url, image_id, alternate_save_path=None):
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
        image_id_fn = image_id.split("/")[-1]
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
        for file_path in glob(os.path.join(download_folder_satname, f'*{image_id_fn}*')):
            if file_path.endswith('.zip'): continue # This is the zip file we took them out of
            short_fn = os.path.basename(file_path)
            # print(short_fn)
            period_split = short_fn.split('.')
            band = period_split[1] # last one is file extention
            short_fn_no_band = period_split[0] # removes extention and band
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
        tiffutils.combine_tiffs(tiff_files=this_image_component_fns) # for each image combine band tiffs into one tiff file
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    

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
                    image_id = image['id']  # Get the ID of the image to download. This is each image not each band 
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
                            download_large_AOI_in_seperate_tiles(sitename=sitename, satname=satname, bands=bands, aoi=aoi, image=image, image_id=image_id)
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

