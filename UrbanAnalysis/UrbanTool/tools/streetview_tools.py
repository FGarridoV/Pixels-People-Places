import time
import os
import requests 
from requests.adapters import HTTPAdapter
#from requests.packages.urllib3.util.retry import Retry
from urllib3.util.retry import Retry
from datetime import datetime
import pandas as pd
import re
import json
from urllib.request import urlopen
import urllib
from urllib.error import HTTPError

from tools.geo_tools import GeoTools as gt 
from shapely.geometry import Point

class StreetViewTools:

    #DELETE IF YOU MAKE IT PUBLIC
    API_KEY = 'INSERT_YOUR_API_KEY_HERE'
    LAST_GRID_QUERIED = -1
    LAST_PANOID_QUERIED = -1

    def get_panoid_from_latlng_google(lat, lng):
        base_link = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
        base_params = 'size=600x300&heading=-45&pitch=42&fov=110&'
        location = f'location={lat},{lng}&'
        key = f'key={StreetViewTools.API_KEY}'
        
        url = base_link + base_params + location + key
        response = urlopen(url)
        data_json = json.loads(response.read())
        
        if data_json['status'] == "ZERO_RESULTS":
            return None
        elif data_json['status'] == "OK":
            pano_id = data_json['pano_id']
            date = data_json['date']
            owner = 'Google' if 'google' in data_json['copyright'].lower() else 'User'
            lat = float(data_json['location']['lat'])
            lng = float(data_json['location']['lng'])

            return [{'panoid': pano_id,
                    'year': int(date.split('-')[0]),
                    'month': int(date.split('-')[1]),
                    'owner': owner,
                    'lat': lat,
                    'lng': lng}]
        

    def _reformat_panos(panos, only_full = True):
        new_panos = []
        for p in panos:
            if 'year' in p.keys() and 'month' in p.keys():
                new_dict = {'panoid': p['panoid'],
                            'year': p['year'],
                            'month': p['month'],
                            'owner': 'Google',
                            'lat': p['lat'],
                            'lng': p['lng']}

                new_panos.append(new_dict)
            else:
                if only_full:
                    continue
                new_dict = {'panoid': p['panoid'],
                            'owner': 'Google',
                            'lat': p['lat'],
                            'lng': p['lng']}
                new_panos.append(new_dict)
        return new_panos


    def panoids_gdf_from_points(query_points, user = False, log = None):
        error = False
        panoid_collector = []
        n = len(query_points)

        try:
            query_points['consulted'] = query_points.apply(lambda row: StreetViewTools._panoids_from_point(row, panoid_collector, n, log, user = user) if row['consulted'] == 0 else 1, axis = 1)
        
        except Exception as e:
            error = True
            last_try = StreetViewTools.LAST_GRID_QUERIED + 1
            if log != None:
                log.write_log(f'An error ocurred in grid point index = {last_try}')
                log.write_error(e)
            total = len(query_points)
            current_status = [1]*last_try + [0]*(total - last_try)
            query_points['consulted'] = pd.Series(current_status)
            if log != None:
                log.write_log(f'Saving current progress')

        if len(panoid_collector)>0:
            panos_gdf = gt.create_gdf_from_panoids(panoid_collector)
            panos_gdf = panos_gdf.drop_duplicates(subset='panoid').reset_index(drop=True)
            panos_gdf['consulted'] = 0
            return panos_gdf, query_points, error
        else:
            return None, query_points, error


    def _add_field(d, field):
        d.update(field)
        return d