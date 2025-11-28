from pathlib import Path
from UrbanTool.city import City
import pandas as pd
import glob

class Project:


    # Constructor to open a city specifying with files you need
    def __init__(self, root_folder, utm = True, log = True, all = False,
                                                            boundary = False, 
                                                            edges = False, 
                                                            nodes = False, 
                                                            grid_points = False, 
                                                            panoids = False, 
                                                            geometries = False,
                                                            detections = False):

        self.root = Path(root_folder)
        self.log = log
        self.utm = utm
        self.all = all
        self.boundary = boundary
        self.edges = edges
        self.nodes = nodes
        self.grid_points = grid_points
        self.panoids = panoids
        self.geometries = geometries
        self.detections = detections


    def prepare_all_cities(self):
        self.cities = {}
        for c in self._get_cities():
            self.cities[c] = City(c, utm = self.utm, log = self.log, all = self.all,
                                                            boundary = self.boundary, 
                                                            edges = self.edges, 
                                                            nodes = self.nodes, 
                                                            grid_points = self.grid_points, 
                                                            panoids = self.panoids, 
                                                            geometries = self.geometries,
                                                            detections = self.detections)

    def _get_cities(self):
        cities = []
        folder = Path(f'{self.root}/*')
        for city in glob.glob(f'{folder}'):
            cities.append(Path(city).parts[-1])
        return cities
    
    def process_images(self):

         for c in self._get_cities():
            self.cities[c] = City(c, utm = self.utm, log = self.log, all = self.all,
                                                            boundary = self.boundary, 
                                                            edges = self.edges, 
                                                            nodes = self.nodes, 
                                                            grid_points = self.grid_points, 
                                                            panoids = self.panoids, 
                                                            geometries = self.geometries,
                                                            detections = self.detections)
    

    def generate_condensed_database(self):
        self.prepare_all_cities()
        dataframes = [] 
        
        for c in self.cities:
            df_aux = self.cities[c].generate_image_database()
            dataframes.append(df_aux)
        
        condensed_db = pd.concat(dataframes)
        condensed_db.reset_index(drop = True, inplace = True)
        condensed_db.to_csv('images.csv')



        




    