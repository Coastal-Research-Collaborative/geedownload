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
            'PAN': 'B8', # this is used for panchromatic sharpening
            'swir1': 'B5',      # SWIR
            'swir2': 'B7',      # SWIR
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Landsat 7 (L7)
        'L7': {
            'B': 'B1',          # Blue
            'G': 'B2',          # Green
            'R': 'B3',          # Red
            'NIR': 'B4', # Near Infrared
            'PAN': 'B8', # this is used for panchromatic sharpening
            'swir1': 'B5',      # SWIR
            'swir2': 'B7',      # SWIR
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Landsat 8 (L8)
        'L8': {
            'B': 'B2',          # Blue
            'G': 'B3',          # Green
            'R': 'B4',          # Red
            'NIR': 'B5', # Near Infrared
            'PAN': 'B8', # this is used for panchromatic sharpening
            'swir1': 'B6',      # SWIR
            'swir2': 'B7',      # SWIR
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Landsat 9 (L9)
        'L9': {
            'B': 'B2',          # Blue
            'G': 'B3',          # Green
            'R': 'B4',          # Red
            'NIR': 'B5', # Near Infrared
            'PAN': 'B8', # this is used for panchromatic sharpening
            'swir1': 'B6',      # SWIR
            'swir2': 'B7',      # SWIR
            'UDM': 'QA_PIXEL'   # QA Band for cloud/shadow
        },
        
        # Sentinel-2 (S2)
        'S2': {
            'B': 'B2',          # Blue
            'G': 'B3',          # Green
            'R': 'B4',          # Red
            'NIR': 'B8', # Near Infrared
            'swir1': 'B11',     # SWIR1
            'swir2': 'B12',     # SWIR2
            'UDM': 'QA10' # past work says this is basically the udm but it says not available'S2Cloudless' # Cloud Mask (using S2Cloudless algorithm)
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
            raise ValueError(f"Invalid channel name '{channel_name}' for satellite '{satname}'")


def retrieve_imagery(sitename, start_date, end_date, data_dir=None, polygon=None, satnames=['L5', 'L7', 'L8', 'L9', 'S2']):
    """
    Download imagery for a given site (if no polygon loads sitename file)

    :param sitename: str the name of the site (used for where the images are downloaded)
    :param start_date: str "YYY-MM-DD" 
    :param end_date: str "YYY-MM-DD" 
    :param data_dir: str directory where the folder (named sitename should be placed)
    :param polygon: 2d list [longitude1, latitude1], [longitude2, latitude2], [longitude3, latitude3], [longitude4, latitude4]] NOTE does not need to be a rectangle
    :param satnames: list of strs the names of the satellites that we want to download imagery from
    """

    authenticate_and_initialize() # authenticate and initialize gee

    if data_dir is  None:
        download_folder = os.path.join('data', 'sat_images', sitename)
    else:
        download_folder = os.path.join(data_dir, 'sat_images', sitename)
    if not os.path.exists(download_folder): os.makedirs(download_folder)

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
    sat_dict = {
        'L5': {'start_year': None, 'end_year': None, 'collection': 'LANDSAT/LT05/C02/T1_TOA'},
        'L7': {'start_year': None, 'end_year': 2022, 'collection': 'LANDSAT/LE07/C02/T1_TOA'},
        'L8': {'start_year': None, 'end_year': None, 'collection': 'LANDSAT/LC08/C02/T1_TOA'},
        'L9': {'start_year': 2022, 'end_year': None, 'collection': 'LANDSAT/LC09/C02/T1_TOA'},
        'S2': {'start_year': None, 'end_year': None, 'collection': 'COPERNICUS/S2_HARMONIZED'}
    }


    for satname in satnames:
        if satname in sat_dict:
            sat_info = sat_dict[satname]
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
            
            
            print(f'These are the bands for {satname}----------------------------------------')
            print(bands)

                

            collection = (ee.ImageCollection(sat_info['collection'])
                          .filterDate(start_date, end_date)
                          .filterBounds(aoi))
            # Check if the collection is not empty
            try:
                n_images = collection.size().getInfo()
            except ee.ee_exception.EEException as e:
                n_images = 0 # if n_images = 0 (it will print out that this is because there are no images avaible)
            if n_images > 0:
                for image in collection.getInfo()['features']:
                    image_id = image['id']  # Get the ID of the image to download
                    print(f"Processing image: {image_id}")

                    image = ee.Image(image_id)

                    scale = image.select(channel_name_to_band('R', satname)).projection().nominalScale().getInfo()
                    print(f'scale of red: {scale}')
                    print('----------------------------------------------------------------')
                    if not 'S' in satname:
                        scale = image.select(channel_name_to_band('PAN', satname)).projection().nominalScale().getInfo()
                        print(f'scale of pancromatic: {scale}')
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
                    download_url = image.getDownloadURL({
                        'scale': 10,
                        'region': aoi.getInfo(),
                        'bands': bands
                    })
                    print(f'Downloading these bands {bands}')
                    print(f"Download URL: {download_url}")

                    # if 'S' in satname:
                    #     # NOTE udm band needs to be removed from bands each itteration because it is added above resampled as udm_resampled
                    #     bands.remove(udm_band)

                    response = requests.get(download_url)

                    # Check if the request was successful (status code 200)
                    if response.status_code == 200:
                        download_folder_satname = os.path.join('data', 'sat_images', sitename, satname) # sitename dir was already made
                        if not os.path.exists(download_folder_satname): os.makedirs(download_folder_satname)

                        # Modify the zip filename to include the satname at the beginning and avoid nested folders
                        image_id_fn = image_id.split("/")[-1]
                        zip_filename = os.path.join(download_folder_satname, f'{image_id_fn}_image.zip')

                        print(zip_filename)
                        
                        # Make sure the download folder exists before saving the file
                        if not os.path.exists(download_folder_satname):
                            os.makedirs(download_folder_satname)
                        
                        with open(zip_filename, 'wb') as f:
                            f.write(response.content)
                        print(f"File downloaded successfully as {zip_filename}")

                        # Unzip the file into the download folder
                        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                            zip_ref.extractall(download_folder_satname)  # Extract directly into the download folder
                        print(f"File unzipped successfully into {download_folder_satname}")

                        # prepend satelite name to file names and replace channel with the actual channel
                        for file_path in glob(os.path.join(download_folder_satname, f'*{image_id_fn}*')):
                            if file_path.endswith('.zip'): continue
                            print(file_path)
                            short_fn = os.path.basename(file_path)
                            # print(short_fn)
                            period_split = short_fn.split('.')
                            band = period_split[1] # last one is file extention
                            short_fn_no_band = period_split[0] # removes extention and band
                            # print(band)
                            short_fn = f'{short_fn_no_band}.{channel_name_to_band(channel_name=band, satname=satname, reverse=True)}'
                            # print(short_fn)

                            new_filename = os.path.join(os.path.dirname(file_path), f"{satname}_{short_fn}.tif")

                            if not file_path == new_filename: os.rename(file_path, new_filename) # NOTE done by resampling for landsay

                        os.remove(zip_filename) # remove zip file
                    else:
                        print(f"Failed to download file. Status code: {response.status_code}")
            else:
                print(f"No images found for {satname} in the given date range and polygon.")


def create_polygon_geojson(sitename:str, coords:list, data_dir:str='data'):
    """
    Given a list of lat long coordinates this creates a polygon function used in the imagery download process
    """
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

    save_dir = os.path.join(data_dir, 'siteinfo', sitename)
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"{sitename}_polygon.geojson")
    
    with open(save_path, 'w') as geojson_file:
        json.dump(geojson_data, geojson_file, indent=4)

