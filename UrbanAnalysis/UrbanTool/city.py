from json import detect_encoding
import warnings
warnings.filterwarnings('ignore', '.*The Shapely GEOS version*', )
import pandas as pd
import geopandas as gpd
from pathlib import Path
import glob
import os

from tools.parallel_tools import Parallel
from tools.logger_tools import Log
from tools import gridgen_tools as gg
from tools import aggregation_tools as at
from tools import model_estimator_tools as met
from tools.zerokchecker_tools import check_and_solve

# Files contained in a City Project
boundary_file = lambda city: Path(f'{city}/boundary/boundary.geojson')
edges_file = lambda city: Path(f'{city}/network/edges/edges.geojson')
nodes_file = lambda city: Path(f'{city}/network/nodes/nodes.geojson')
grid_points_file = lambda city: Path(f'{city}/grid_points/grid_points.geojson')
panoids_file = lambda city: Path(f'{city}/panoids/panoids.geojson')
graph_file = lambda city: Path(f'{city}/network/graph/graph.gpickle')
geometry_file = lambda city, geo: Path(f'{city}/geometries/{geo}/{geo}.geojson')
detections_file = lambda city: Path(f'{city}/detections/detections.geojson')
cells_file = lambda city: Path(f'{city}/cells/cells.geojson')
cells_agg_file = lambda city: Path(f'{city}/cells/cells_agg.geojson')


class City:

    DETECTED_OBJ = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 
                'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                'parking meter', 'bench']

    # Constructor to open a city specifying with files you need
    def __init__(self, zone_folder, utm = True, log = True, abs_path = False,  log_code = "temp",
                                                                                all = False,
                                                                                boundary = False, 
                                                                                edges = False, 
                                                                                nodes = False, 
                                                                                grid_points = False, 
                                                                                panoids = False, 
                                                                                geometries = False,
                                                                                detections = False,
                                                                                cells = False,
                                                                                cells_agg = False):
        # Main city variables 
        self.folder = None
        self.status = None
        self.logger = None
        self.utm = None
        self.crs = None

        self.boundary = None
        self.edges = None
        self.nodes = None
        self.grid_points = None
        self.panoids = None
        self.detections = None
        self.cells = None
        self.cells_agg = None

        self.boundary_utm = None
        self.edges_utm = None
        self.nodes_utm = None
        self.grid_points_utm = None
        self.panoids_utm = None
        self.detections_utm = None
        self.cells_utm = None
        self.cells_agg_utm = None

        
        # Setting the city folder 
        if abs_path:
            self.folder = Path(f'{zone_folder}')
        else:
            self.folder = Path(f'zones/{zone_folder}')

        # Working with UTM
        self.utm = utm


        # Logger creation
        if log:
            self.logger = Log(f'{self.folder}', add = f'_{log_code}')
        else:
            self.logger = None
        

        # Open the CRS files
        with open(Path(f'{self.folder}/crs.txt'), 'r') as f:
            self.crs = int(f.read())
        
        # Checking if the selected city is already completed
        if abs_path:
            self.status = True
        else: 
            with open(Path(f'{self.folder}/log.txt'), 'r') as f:
                verifier = f.readlines()[-1].split('-')[-1].strip().split(' ')[-1]
                self.status = True if verifier == 'successfully' or verifier == 'succesfully' else False

       # Creation of variables
        if all:
            boundary, edges, nodes, grid_points, panoids, geometries = [True]*6

        # Getting files if where indicated
        if self.status:
            if boundary or all:
                self.boundary = gpd.read_file(boundary_file(self.folder))
                if utm:
                    self.boundary_utm = self.boundary.to_crs(self.crs)
            
            if edges:
                self.edges = gpd.read_file(edges_file(self.folder))
                if utm:
                    self.edges_utm = self.edges.to_crs(self.crs)
            
            if nodes:
                self.nodes = gpd.read_file(nodes_file(self.folder))
                if utm:
                    self.nodes_utm = self.nodes.to_crs(self.crs)
            
            if grid_points:
                self.grid_points = gpd.read_file(grid_points_file(self.folder))
                if utm:
                    self.grid_points_utm = self.grid_points.to_crs(self.crs)
            
            if panoids:
                self.panoids = gpd.read_file(panoids_file(self.folder))
                if utm:
                    self.panoids_utm = self.panoids.to_crs(self.crs)

            if geometries:
                for g in self._get_geometries():
                    setattr(self, g, gpd.read_file(geometry_file(self.folder,g)))
                    if utm:
                        setattr(self, f'{g}_utm', gpd.read_file(geometry_file(self.folder,g)).to_crs(self.crs))
            
            if detections and os.path.exists(f'{self.folder}/detections'):
                self.detections = gpd.read_file(detections_file(self.folder))
                for c in City.DETECTED_OBJ:
                    self.detections = self.detections.rename(columns = {c: f'{c}_det'})
                if utm:
                    self.detections_utm = self.detections.to_crs(self.crs)
            
            if cells and os.path.exists(f'{self.folder}/cells'):
                self.cells = gpd.read_file(cells_file(self.folder))
                if utm:
                    self.cells_utm = self.cells.to_crs(self.crs)
            
            if cells_agg and os.path.exists(f'{self.folder}/cells'):
                self.cells_agg = gpd.read_file(cells_file(self.folder))
                if utm:
                    self.cells_agg_utm = self.cells_agg.to_crs(self.crs)
    
    # New folder that needs to be created
    def _creates_detection_folder(self):
        detection_folder = Path(f'{self.folder}/detections')
        if not os.path.exists(f'{detection_folder}'):
                os.mkdir(f'{detection_folder}')
        return detection_folder
    
    def _creates_cells_folder(self):
        cells_folder = Path(f'{self.folder}/cells')
        if not os.path.exists(f'{cells_folder}'):
                os.mkdir(f'{cells_folder}')
        return cells_folder
            
    def _get_geometries(self):
        geos = []
        geometries_folder = Path(f'{self.folder}/geometries/*')
        for geo in glob.glob(f'{geometries_folder}'):
            geos.append(Path(geo).parts[-1])
        return geos
    
    def _apply_split_images(row, db_collector):
        panoid = row['panoid']
        link_a = row['url_side_a']
        link_b = row['url_side_b']
        year = row['year']
        city = row['city']
        lat = row['lat']
        lng = row['lng']

        rows = [{'image_id': panoid + 'a', 'url': link_a, 'year': year, 'city': city, 'lat': lat, 'lng': lng},
            {'image_id': panoid + 'b', 'url': link_b, 'year': year, 'city': city, 'lat': lat, 'lng': lng}]
        df = pd.DataFrame(rows)
        db_collector.append(df)
    
    # Extra methods
    def downloaded_images_were_checked(self):
        checked_path = Path(f'{self.folder}/checked')
        return os.path.exists(f'{checked_path}')
    
    def check_downloads(self):
        name = f'{self.folder}'.split('/')[1]
        check_and_solve(name, self.panoids)

    ## Detect objects on images (requires panoids)
    def detect_objects(self, cores = 0, model = 'ssd3', sub_classes = True, extras = 0):

        current_folder = self._creates_detection_folder()
        detections = self.panoids.copy()

        n = len(detections)
        imagedb = Path(f"{self.folder}/imagedb")

        #model, im_folder, subclasses
        if sub_classes:
            sc = 'True'
        else:
            sc = 'False'
        params = (model, f"{imagedb}", sc)

        process = Parallel(detections, current_folder, params, script = 'image_detection' )
        detections = process.paralelize_execution(cores = cores + extras)
        
        self.detections = detections.copy()
        self.detections.to_file(f'{self.folder}/detections/detections.geojson')
        if self.utm:
            self.detections_utm = self.detections.to_crs(self.crs)


    ## Aggregate the information (requieres detections, boundary and should be UTM)
    def aggregate_info_in_cells(self, min_year, max_year, point_list, polygon_list, side = 50):
        self._creates_cells_folder()
        cells = gg.create_grid_on_gdf(self.boundary_utm, shape = 'hexagon', side = side)

        # Aggregation of detections
        cells = at.aggregate_detections_on_cells(self.detections_utm, cells, City.DETECTED_OBJ, min_year, max_year, criteria = 'avg')  

        # Aggregation of network
        cells = at.aggregate_network_info_on_cells(self.edges_utm, self.nodes_utm, cells)

        # Aggregation of amenities
        for p in point_list:
            df = p[0]
            col = p[1]
            cells = at.aggregate_point_info_on_cells(df, cells, p[1])

        # Aggregation of landuses
        for p in polygon_list:
            df = p[0]
            col = p[1]
            cells = at.aggregate_polygon_info_on_cells(df, cells, p[1])
        
        self.cells = cells.to_crs(4326)
        self.cells.to_file(f'{self.folder}/cells/cells.geojson')
        if self.utm:
            self.cells_utm = self.cells.to_crs(self.crs)

    # Aggregate information contained in cells (requires cells)        
    def get_cells_aggregated(self):
        self.cells_agg = self.cells.copy()
        for c in City.agg_amn_dict:
            new_col = f"{c}_amn_agg"         
            self.cells_agg[new_col] = 0

            for sub_c in City.agg_amn_dict[c]:
                if sub_c in self.cells_agg.columns:
                    self.cells_agg[new_col] = self.cells_agg[new_col] + self.cells_agg[sub_c]
                    self.cells_agg.drop(columns = sub_c)
        self.cells_agg.to_file(f'{self.folder}/cells/cells_agg.geojson')
        if self.utm:
            self.cells_agg_utm = self.cells_agg.to_crs(self.crs)
        
    # Estimate a model types: nrm, agg, std    
    def estimate_ols_model(self, y_col, X_cols, kind = 'agg'):

        if kind == 'nrm':
            self.cell_for_estimation = self.cells.loc[self.cells['has_data']==1].copy()
            self.nrm_model = met.estimate_ols_model(self.cell_for_estimation, y_col, X_cols, add_constant = True, standarized = False)
        elif kind == 'agg':
            self.get_cells_aggregated()
            self.cell_for_estimation_agg = self.cells_agg.loc[self.cells_agg['has_data']==1].copy()
            self.agg_model = met.estimate_ols_model(self.cell_for_estimation_agg, y_col, X_cols, add_constant = True, standarized = False)
        elif kind == 'std':
            self.get_cells_aggregated()
            self.cell_for_estimation_agg = self.cells_agg.loc[self.cells_agg['has_data']==1].copy()
            self.std_model = met.estimate_ols_model(self.cell_for_estimation_agg, y_col, X_cols, add_constant = False, standarized = True)

    #######
    ## Generates a image database (requires panoids)
    def generate_image_database(self, from_year = 2019):
        db = self.panoids.copy()

        db = self.panoids.loc[self.panoids['year'] >= from_year]
        name = str(self.folder).split('/')[1].split('_')[0]

        db = db.loc[:,['panoid', 'url_side_a', 'url_side_b', 'year', 'lat', 'lng']]
        db['city'] = name

        db_collector = []
        db.apply(lambda row: City._apply_split_images(row, db_collector), axis = 1)

        df_final = pd.concat(db_collector)
        df_final.reset_index(drop = True, inplace = True)
        return df_final


    agg_amn_dict = {'food_place': ['bar_amn', 'biergarten_amn', 'cafe_amn', 
                                   'fast_food_amn', 'food_court_amn', 
                                   'ice_cream_amn', 'pub_amn', 'restaurant_amn'],
                    'education': ['college_amn', 'driving_school_amn', 
                                  'kindergarten_amn', 'language_school_amn', 
                                  'library_amn', 'toy_library_amn',
                                  'music_school_amn', 'school_amn', 'university_amn'],
                    'transportation': ['bicycle_parking_amn', 'bicycle_repair_station_amn',
                                       'bicycle_rental_amn', 'boat_rental_amn', 
                                       'boat_sharing_amn', 'bus_station_amn', 'car_rental_amn',
                                       'car_sharing_amn', 'car_wash_amn', 'vehicle_inspection_amn',
                                       'charging_station_amn', 'ferry_terminal_amn', 'fuel_amn', 
                                       'grit_bin_amn', 'motorcycle_parking_amn', 'parking_amn',
                                       'parking_entrance_amn', 'parking_space_amn', 'taxi_amn'],
                    'financial': ['atm_amn', 'bank_amn', 'bureau_de_change_amn', 'Healthcare_amn',
                                  'baby_hatch_amn', 'clinic_amn', 'dentist_amn', 'doctors_amn',
                                  'hospital_amn', 'nursing_home_amn', 'pharmacy_amn', 
                                  'social_facility_amn', 'veterinary_amn'],
                    'entertainment': ['arts_centre_amn', 'brothel_amn', 'casino_amn', 'cinema_amn', 
                                      'community_centre_amn', 'conference_centre_amn', 'events_venue_amn',
                                      'fountain_amn', 'gambling_amn', 'love_hotel_amn', 'nightclub_amn',
                                      'planetarium_amn', 'public_bookcase_amn', 'social_centre_amn',
                                      'stripclub_amn', 'studio_amn', 'swingerclub_amn', 'theatre_amn'],
                    'public_service': ['courthouse_amn', 'fire_station_amn', 'police_amn', 'post_box_amn',
                                       'post_depot_amn', 'post_office_amn', 'prison_amn', 'ranger_station_amn',
                                       'townhall_amn'],
                    'facility': ['bbq_amn', 'bench_amn', 'dog_toilet_amn', 'drinking_water_amn', 'give_box_amn', 
                                 'parcel_locker_amn', 'shelter_amn', 'shower_amn', 'telephone_amn', 'toilets_amn', 
                                 'water_point_amn', 'watering_place_amn'],
                    'waste': ['sanitary_dump_station_amn', 'recycling_amn', 'waste_basket_amn', 'waste_disposal_amn',
                              'waste_transfer_station_amn'],
                    'other': ['animal_boarding_amn', 'animal_breeding_amn', 'animal_shelter_amn', 'baking_oven_amn', 
                              'childcare_amn', 'clock_amn', 'crematorium_amn', 'dive_centre_amn', 'funeral_hall_amn', 
                              'grave_yard_amn', 'hunting_stand_amn', 'internet_cafe_amn', 'kitchen_amn', 'kneipp_water_cure_amn', 
                              'lounger_amn', 'marketplace_amn', 'monastery_amn', 'photo_booth_amn', 'place_of_mourning_amn', 
                              'place_of_worship_amn', 'public_bath_amn', 'public_building_amn', 'refugee_site_amn', 
                              'vending_machine_amn']}
    
    agg_lu_dict = {'commercial': ['commercial'],
                   'construction': ['construction'],
                   'education': ['education'],
                   'industrial': ['industrial'],                  
                   'residential': ['residential'],
                   'retail': ['retail'],
                   'institutional': ['institutional'],
                   'rural': ['allotments', 'farmland', 'farmyard', 'flowerbed', 'orchard', 'vineyard', 'greenhouse_horticulture',
                             'plant_nursery'],
                   'natural': ['grass', 'forest', 'meadow', 'conservation'],
                   'water': ['aquaculture', 'basin', 'reservoir', 'salt_pond']
                   }





#brownfield
#greenfield
#cemetery
#depot
#garages
#landfill
#military
#port
#quarry
#railway
#recreation_ground
#religious
#village_green
#winter_sports
#user defined








        











