# geedownload
`geedownload` is a Python package with functions to facilitate downloading landsat and sentinel imagery from Google Earth Engine

First step:
`pip install git+https://github.com/Coastal-Research-Collaborative/geedownload.git`

Then use like this:

import geedownload

sitename = 'georgiajekyllisland'

coords = [
    [-81.41396967622494,31.035661672924554],
    [-81.40667406770443,31.035661672924554],
    [-81.40667406770443,31.053126706868298],
    [-81.41396967622494,31.053126706868298],
    [-81.41396967622494,31.035661672924554]
]

geedownload.create_polygon_geojson(sitename, coords = coords)

start_date = '2024-07-01'
end_date = '2024-08-30'

geedownload.retrieve_imagery(sitename=sitename, 
                          start_date = start_date,
                          end_date = end_date,
                          data_dir  = os.path.join('data', 'sat_images'),
                          polygon = coords
)
