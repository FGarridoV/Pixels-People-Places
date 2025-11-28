from asyncio.log import logger
import os
import glob
import pickle
import pandas as pd

import warnings
warnings.filterwarnings('ignore', '.*The Shapely GEOS version*', )
warnings.filterwarnings('ignore', '.*__len__ for multi-part geometries*', )
warnings.filterwarnings('ignore', '.*Iteration over multi-part geometries*', )

import geopandas as gpd
import networkx as nx

from tools.logger_tools import Log
from tools.geo_tools import GeoTools as gt
from tools.streetview_tools import StreetViewTools as svt
from tools.zerokchecker_tools import check_and_solve

class Zone:

    def __init__(self, name, region, country, country_code, timeout = 600, 
                 logs = True):

        #Restaring status
        svt.LAST_GRID_QUERIED = -1
        svt.LAST_PANOID_QUERIED = -1

        # Variables
        self.folder = f'zones/{name}_{country_code}'
        self.logs = logs
        self.logger = None

        self.name = name
        self.region = region
        self.country = country
        self.crs = gt.get_utm_epsg_from_place(f'{name}, {country}')

        self.boundary = None
        self.graph = None
        self.edges = None
        self.nodes = None
        self.grid_points = None
        self.panoids = None
        self.geometries = {}

        self.create_zones_folders()
        gt.timeout = timeout
        if self.logs:
            self.logger = Log(self.folder)

    @property
    def boundary_utm(self):
        return self.boundary.to_crs(self.crs)
    
    @property
    def edges_utm(self):
        return self.boundary.to_crs(self.crs)

    @property
    def nodes_utm(self):
        return self.boundary.to_crs(self.crs)

    @property
    def grid_points_utm(self):
        return self.boundary.to_crs(self.crs)

    @property
    def panoids_utm(self):
        return self.boundary.to_crs(self.crs)

    @property
    def geometries_utm(self):
        geometries = {}
        for k in self.tags:
            geometries.update({k: self.geometries[k].to_crs(self.crs)})
        return geometries


    def create_zones_folders(self):

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        sub_folders = ['boundary', 'network', 'network/graph', 'network/edges',
                       'network/nodes', 'geometries', 'grid_points', 'panoids',
                       'imagedb']
        
        for sf in sub_folders:
            if not os.path.exists(f'{self.folder}/{sf}'):
                os.mkdir(f'{self.folder}/{sf}')

        f = open(f'{self.folder}/crs.txt', 'w')
        f.write(str(self.crs))
        f.close()


    def set_boundary(self, boundary_opt, polygon = None, ext = 'geojson', accuracy = False, save = True):

        if os.path.exists(f'{self.folder}/boundary/boundary.{ext}'):
            if self.logs:
                self.logger.write_log(f'A boundary file found for {self.name}')
            boundary_opt = 'current'

        location = f'{self.name}, {self.country}'

        if boundary_opt == 'region': 
            if self.logs:
                self.logger.write_log(f'Getting boundary for {self.name} based on place')
            self.boundary = gt.gdf_polygon_based_on_place(location, accuracy = accuracy)

        elif boundary_opt == 'select':
            if self.logs:
                self.logger.write_log(f'Getting boundary for {self.name} based on a selected area')
            self.boundary = gt.gdf_polygon_based_on_draw(location)
        
        elif boundary_opt == 'polygon':
            if self.logs:
                self.logger.write_log(f'Getting boundary for {self.name} based on given polygon')
            self.boundary = gt.create_gdf_from_polygon(polygon)

        elif boundary_opt == 'current':
            if self.logs:
                self.logger.write_log(f'Reading current boundary for {self.name}')
            folder = f'{self.folder}/boundary'
            boundary_path = glob.glob(f'{folder}/*.{ext}')[0]
            self.boundary = gpd.read_file(boundary_path)
        
        elif 'addr:' in boundary_opt:
            if self.logs:
                self.logger.write_log(f'Reading current boundary from address for {self.name}')
                address = boundary_opt.split('addr:')[-1]
            self.boundary = gt.gdf_polygon_based_on_address(address)

        else:
            self.boundary = None

        if save:
            self._save_gis('boundary')
            if self.logs:
                self.logger.write_log(f'boundary saved for {self.name}')
    

    # Method to download or open all network related files (gdf edges, 
    # gdf nodes and graph network) and stored in zone object.
    # Inputs
    ## network_type: str = 
    ## street_types: str = 
    ## simplify: bool = 
    ## crop: bool =
    ## save: bool = 
    def set_network(self, network_type, street_types, simplify = True, 
                    crop = True, save = True):

        edges_file = f'{self.folder}/network/edges/edges.geojson'
        nodes_file = f'{self.folder}/network/nodes/nodes.geojson'
        graph_file = f'{self.folder}/network/graph/graph.gpickle'

        are_edges = os.path.exists(edges_file)
        are_nodes = os.path.exists(nodes_file)
        are_graph = os.path.exists(graph_file)

        if are_edges and are_nodes and are_graph:
            if self.logs:
                self.logger.write_log(f'Network files found for {self.name}')
            
            self._open_gis('edges', ext = 'geojson')
            self._open_gis('nodes', ext = 'geojson')
            self._open_graph()
            
            if self.logs:
                self.logger.write_log(f'Network files opened for {self.name}')
        
        else:
            if self.logs:
                self.logger.write_log(f'Getting network for {self.name}')

            boundary = self.boundary.unary_union

            graph, edges, nodes = gt.network_based_on_polygon(boundary, network_type = network_type, street_types = street_types, simplify = simplify)
            self.graph = graph

            if not edges.empty and not nodes.empty:
                if crop == True:
                    edges = edges.clip(self.boundary)
                    edges = edges.explode(index_parts = True)
                    nodes = nodes.clip(self.boundary)
                self.edges = edges.copy()
                self.nodes = nodes.copy()

                if save:
                    self._save_gis('edges')
                    self._save_gis('nodes')
                    self._save_graph()
                    if self.logs:
                        self.logger.write_log(f'Network graph, edges and nodes saved for {self.name}')

            else:
                if self.logs:
                    self.logger.write_log(f'404: Network NOT FOUND')


    # Download the subgeometries specified by the tags
    def new_geometry(self, name, tags, kind = 'any', and_tags = None, crop = True, save = True):

        if os.path.exists(f'{self.folder}/geometries/{name}/{name}.geojson'):
            
            if self.logs:
                self.logger.write_log(f'Tag {name} file found for {self.name}')
            
            self._open_gis(name, ext = 'geojson')
            
            if self.logs:
                self.logger.write_log(f'Tag {name} file opened for {self.name}')

        else:
            if self.logs:
                self.logger.write_log(f'Getting {kind} shapes of tag {name} for {self.name}')

            boundary = self.boundary.unary_union

            if not and_tags is None:
                tags.update(and_tags)

            gdf = gt.geometries_based_on_polygon(tags, boundary, kind)

            if not and_tags is None:
                cond = (True)
                for c in and_tags:
                    cond = cond and (gdf[c] == and_tags[c])
                gdf = gdf.loc[cond]

            if not gdf.empty:
                if crop == True:
                    gdf = gdf.clip(self.boundary)
                self.geometries[name] = gdf.copy()

                if save:
                    self._save_gis(name)
                    if self.logs:
                        self.logger.write_log(f'Tags {name} saved for {self.name}')
            else:
                if self.logs:
                    self.logger.write_log(f'404: Tag {name} NOT FOUND')


    # Method X used to generate a grid of points based on an specific geometry
    # Inputs
    ## gap: int = 
    ## key_crop: bool = 
    ## save: bool =
    ## new: bool =
    def set_grid_points(self, gap = 30, key_crop = 'boundary', save = True):
        
        if os.path.exists(f'{self.folder}/grid_points/grid_points.geojson'):
            self._open_gis('grid_points', ext = 'geojson')
            
            if self.logs:
                self.logger.write_log(f'Opened current grid_points for {self.name}')
        else:
            if self.logs:
                self.logger.write_log(f'Creating grid points with {gap} meters within {key_crop} for {self.name}')

            geometry = self.boundary
            gdf = gt.grid_points_based_on_polygon(geometry, gap, utm = self.crs)

            if not gdf.empty:
                if key_crop != 'boundary':
                    gdf_crop = self.geometries[key_crop]
                    gdf = gdf.clip(gdf_crop)
                    gdf = gdf.reset_index(drop = True)

            name = 'grid_points'
            self._add_gis(gdf, name, save = save)
    

    # Method X used to collect panoids from Google based on the current grid points
    # Inputs
    ## query_points: str = 
    ## user: bool = 
    def get_panoids(self, user = False):
        if self.logs:
            self.logger.write_log(f'Getting panoids based on grid_points for {self.name}')
        
        are_panoids = os.path.exists(f'{self.folder}/panoids/panoids.geojson')
        points = self.grid_points
        panoids, new_points, error = svt.panoids_gdf_from_points(points, user = user, log = self.logger)

        # All panoids were collected
        if panoids is None and are_panoids:
            if self.logs:
                self.logger.write_log(f'Opening panoids from existing file in {self.name}')

            self._open_gis('panoids', ext= 'geojson')
        
        # New panoids have found
        elif type(panoids) is gpd.GeoDataFrame and panoids.empty == False:
            
            # There are previous panoids
            if are_panoids:
                if self.logs:
                    self.logger.write_log(f'Adding new panoids found in {self.name}')
            
                self._open_gis('panoids', ext= 'geojson')
                self.panoids = pd.concat([self.panoids, panoids])
                self.panoids.drop_duplicates(subset='panoid', inplace = True)
                self.panoids.reset_index(drop=True, inplace = True)
                self._save_gis('panoids')
            
            # First time collecting panoids
            else:
                if self.logs:
                    self.logger.write_log(f'Saving panoids found in {self.name}')

                self._add_gis(panoids, 'panoids', ext = 'geojson', save = True)

        if not error and self.logs:
            self.logger.write_log(f'Collected all panoids for {self.name}')

        name = f'grid_points'
        self._add_gis(new_points, name, save = True)
        if self.logs:
                self.logger.write_log(f'Updated grid_points status in {self.name}')

        if error:
            return -1
        else:
            return 200


    # Method X used to collect Google Street View images based on the current panoids
    # Inputs
    ## max_dist: int = 
    def collect_street_view_images(self, max_dist = None, url_only = False):
        if not 'angle' in self.panoids.columns.to_list():
            self._add_angles_to_panoids(max_dist = max_dist)
        panoids = self.panoids.copy()
        img_folder = f'{self.folder}/imagedb'
        panoids_w_images, was_error = svt.add_images_to_panoids_db(panoids, img_folder, url_only = url_only, log = self.logger)
        self._add_gis(panoids_w_images, 'panoids')
        if was_error:
            return -1
        else:
            return 200
    
    def check_downloads(self):
        name = self.folder.split('/')[1]
        check_and_solve(name, self.panoids)

    
    # Internal method used to measure the angles of the street associated to 
    # each pano
    # Inputs
    ## max_dist: int = 
    def _add_angles_to_panoids(self, buffer = 2, max_dist = None):
        if self.logs:
            self.logger.write_log(f'Measuring angles of links for {self.name}')

        # Get panoids and edges in UTM standars
        panoids = self.panoids.to_crs(self.crs)
        edges = self.edges.to_crs(self.crs).reset_index(drop=True)

        # For each panoid get nearest link
        nearests = gpd.sjoin_nearest(panoids, edges, how = 'left', distance_col = 'dist', rsuffix = 'edge', max_distance = max_dist)
        nearests.drop_duplicates(subset = 'panoid', keep = 'first', inplace = True)
        nearests = nearests.join(edges['geometry'], how = 'left', on = 'index_edge', rsuffix = '_edge')

        # Get angle of each link
        nearests['angle'] = nearests.apply(lambda row: gt.get_angle_of_line(row['geometry_edge'],row['geometry'], row['dist'], buffer = buffer), axis = 1)

        # Add data to panoids
        new_panoids = panoids.merge(nearests[['panoid','dist','angle']], how = 'left', on = 'panoid')
        new_panoids = new_panoids.to_crs(4326)
        self._add_gis(new_panoids, 'panoids')

        if self.logs:
            self.logger.write_log(f'Angles of links measured for {self.name}')

    
    # Internal method used to store a new gdf into the variables of Zone object
    # Inputs
    ## gdf: GeoDataFrame = 
    ## name: str = 
    ## ext: str = 
    ## save: bool = 
    def _add_gis(self, gdf, name, ext = 'geojson', save = True):
        if name in ['edges', 'nodes', 'boundary', 'grid_points', 'panoids']:
            setattr(self, name, gdf.copy())
        else:
            self.geometries[name] = gdf.copy()
        if save:
            self._save_gis(name, ext)
    

    # Internal method used to save all GIS content generated by the software
    # Inputs
    ## name: str = 
    ## ext: str = 
    def _save_gis(self, name, ext = 'geojson'):
        if name in ['edges', 'nodes']:
            file_path = f'{self.folder}/network/{name}/{name}.{ext}'
            getattr(self, name).to_file(file_path)
        elif name in ['boundary', 'grid_points', 'panoids']:
            file_path = f'{self.folder}/{name}/{name}.{ext}'
            getattr(self, name).to_file(file_path)
        else:
            folder = f'{self.folder}/geometries/{name}'
            if not os.path.exists(folder):
                os.mkdir(folder) 
            file_path = f'{folder}/{name}.{ext}'
            gdf = self.geometries[name]
            gdf.to_file(file_path)
    

    # Internal method used to save the networkx graph generated by the software
    # Inputs
    ## ext: str = 
    def _save_graph(self, ext = 'gpickle'):
        file_path = open(f'{self.folder}/network/graph/graph.{ext}', 'wb')
        pickle.dump(self.graph, file_path, pickle.HIGHEST_PROTOCOL)
        file_path.close()
        

    # Internal method used to read geographic gdf files and store it into the 
    # variables of Zone object.
    # Inputs
    ## name: str = 
    ## ext: str = 
    def _open_gis(self, name, ext = 'geojson'):
        if name in ['edges', 'nodes']:
            file_path = f'{self.folder}/network/{name}/{name}.{ext}'
            gdf = gpd.read_file(file_path)
            setattr(self, name, gdf)
        elif name in ['boundary', 'grid_points', 'panoids']:
            file_path = f'{self.folder}/{name}/{name}.{ext}'
            gdf = gpd.read_file(file_path)
            setattr(self, name, gdf)
        else:
            file_path = f'{self.folder}/geometries/{name}/{name}.{ext}'
            gdf = gpd.read_file(file_path)
            self.geometries[name] = gdf

    # Internal method used to open the networkx graph from file to Zone obj
    # Inputs
    ## ext: str = 
    def _open_graph(self):
        file_path = open(f'{self.folder}/network/graph/graph.gpickle', 'rb')
        self.graph = pickle.load(file_path)
        file_path.close()