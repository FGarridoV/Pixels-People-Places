# Run Data Collection one city
from zone import Zone

import warnings
warnings.filterwarnings('ignore', '.*The Shapely GEOS version*', )
warnings.filterwarnings('ignore', '.*__len__ for multi-part geometries*', )
warnings.filterwarnings('ignore', '.*Iteration over multi-part geometries*', )

# SCRIPT

## Params
name = 'Delft'
region = None # or 'South Holland'
country = 'the Netherlands'
country_code = 'NL'
logs = True

## Tags
tags = [{'name': 'amenities', 'kind': 'point', 'tag': {'amenity': True}},
        {'name': 'landuses', 'kind': 'polygon', 'tag': {'landuse': True}}]

# OPTIONS
boundary_opt = 'region' # or 'region', 'select', 'current', 'polygon'
network_type = 'all' # or 'all_private', 'all', 'bike', 'drive', 'drive_service', 'walk'
street_types = 'all' # or 'all' ['primary', 'secondary', 'terciary']
gap_grid = 200 # meters
zone_grid = 'boundary' # or any polygon tag

city_accuracy = True
download_images = True

### Creates the zone class
zone = Zone(name, region, country, country_code, logs = logs)

### Creates the zones folders
zone.create_zones_folders()

### Set boundary
zone.set_boundary(boundary_opt, accuracy = city_accuracy)

### Set network
zone.set_network(network_type, street_types, simplify = True)

### Add geometries
for t in tags:
    zone.new_geometry(t['name'], t['tag'], t['kind'], and_tags = None)

### Set grid points
zone.set_grid_points(gap = gap_grid, key_crop = zone_grid)

### Get panoids from gridpoints
res1 = zone.get_panoids(user = False)
if res1 == -1:
    zone.logger.write_log(f'{name} Collection failed (panoid collection)')

### Get collects GSV
res2 = zone.collect_street_view_images(url_only = not download_images)
if res2 == -1:
    zone.logger.write_log(f'{name} Collection failed (images collection)')

if res1 != -1 and res2 != -1:
    zone.logger.write_log(f'{name} Collection runs successfully')
