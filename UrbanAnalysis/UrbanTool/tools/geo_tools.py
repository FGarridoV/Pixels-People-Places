import re, utm, math, time, requests, logging
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from waiting import wait
from collections import OrderedDict
from shapely.geometry import Point, LineString, Polygon
from tools.interactive_map import InteractiveMap

import warnings
warnings.filterwarnings('ignore', '.*__len__ for multi-part geometries*', )
warnings.filterwarnings('ignore', '.*Iteration over multi-part geometries*', )

class GeoTools:

    timeout = 600

    # Returns a Polygon objects and a GeoDataFrame based on a list of points
    # output: (Polygon, GeoDataFrame)
    def create_gdf_from_points(points):
        polygon = Polygon(points)
        data = {'zone': [1], 'geometry': [polygon]}
        gdf = gpd.GeoDataFrame(data, crs = 4326)
        return polygon, gdf


    # Returns a Polygon objects and a GeoDataFrame based on a list of points
    # output: (Polygon, GeoDataFrame)
    def create_gdf_from_polygon(polygon, crs = 4326):
        data = {'zone': [1], 'geometry': [polygon]}
        gdf = gpd.GeoDataFrame(data, crs = crs)
        return gdf


    # Returns a polygon GeoDataFrame based on a place.
    # output: GeoDataFrame
    def gdf_polygon_based_on_place(place, clean = True, accuracy = False):
        ox.settings.timeout = GeoTools.timeout
        if accuracy:
            gdf_area = GeoTools.gdf_polygon_based_on_place_city_nl(place)
            if gdf_area is None:
                gdf_area = ox.geocoder.geocode_to_gdf(place)
        else:
            gdf_area = ox.geocoder.geocode_to_gdf(place)
        if clean:
            gdf_area = GeoTools._simplify_gdf_columns(gdf_area)
        return gdf_area
    
    # Returns a polygon GeoDataFrame based on address
    # output: GeoDataFrame
    def gdf_polygon_based_on_address(address, radius = 800):
        ox.settings.timeout = GeoTools.timeout
        coords = ox.geocoder.geocode(address)
        point = gpd.GeoDataFrame({'name': ['center'], 'geometry': [Point(coords[1], coords[0])]}, crs = 4326).to_crs(28992)
        point['geometry'] = point.buffer(radius)
        gdf_area = point.to_crs(4326)
        return gdf_area


    # Returns a polygon GeoDataFrame based on a place.
    # output: GeoDataFrame
    def gdf_polygon_based_on_place_city_nl(place):
        ox.settings.timeout = GeoTools.timeout

        params = OrderedDict()
        params["q"] = place
        params["format"] = "geojson"
        params["polygon_geojson"] = 1
        params["limit"] = 10

        results = GeoTools.nominatim_request(params)['features']
        found = 0
        for r in results:
            if r['properties']['place_rank'] == 14:
                found = 1
                return gpd.GeoDataFrame.from_features([r], crs = 4326)
        if found == 0:
            return None
        

    # Launch an interactive applet to select an area to study
    # A lat,lng or a place can be given to center the place
    # output: GeoDataFrame
    def gdf_polygon_based_on_draw(place = None, lat = None, lng = None):
        ox.config(timeout=GeoTools.timeout)

        if lat == None or lng == None:
            lat, lng = ox.geocoder.geocode(place)

        map_center = {'lon': lng, 'lat': lat }
        map_selector = InteractiveMap(map_center)
        map_selector.run_app()
        wait(lambda: map_selector.study_area_points != None, timeout_seconds=120, 
            waiting_for="something to be ready")
        map_selector.close_app()
        boundary_points = map_selector.study_area_points
        _, gdf_area = GeoTools.create_gdf_from_points(boundary_points)
        return gdf_area


    # Return a Nx netowrk, edges Gdf and nodes gdf
    def network_based_on_polygon(polygon, network_type = 'all', street_types = 'all', simplify = True):
        ox.settings.timeout = GeoTools.timeout
        
        if street_types == 'all':
            custom_filter = None
        else:
            sep = '|'
            custom_filter = f'["highway"~"{sep.join(street_types)}"]'

        if polygon.geom_type == 'Polygon':
            G = ox.graph.graph_from_polygon(polygon, network_type = network_type, simplify = True, custom_filter=custom_filter)
            G = ox.utils_graph.get_undirected(G)
        elif polygon.geom_type == 'MultiPolygon':
            nets = []
            for p in polygon:
                G_aux = ox.graph.graph_from_polygon(p, network_type = network_type, simplify = True, custom_filter=custom_filter)
                G_aux = ox.utils_graph.get_undirected(G_aux)
                nets.append(G_aux)
            G = nx.compose(*nets)

        network = G.copy()
        nodes = ox.graph_to_gdfs(G, edges=False)
        edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
        nodes.rename(columns={"street_count": "st_count"}, inplace = True)
        edges = edges.loc[:,['u', 'v', 'key', 'geometry']]
        edges = edges.explode(index_parts=True).reset_index(drop = True)
        return network, edges, nodes
        

    # Return a GeoDataFrame of geometries from a given polygon based on tags
    # type = ['any', 'point', 'line', 'polygon']
    def geometries_based_on_polygon(tags, polygon, format_cols = True, kind = 'any'):
        ox.settings.timeout = GeoTools.timeout

        gdf = ox.geometries.geometries_from_polygon(polygon, tags)

        if format_cols:
            gdf = GeoTools._simplify_gdf_columns(gdf)

        if not gdf.empty:
            if kind == 'any':
                gdf = GeoTools._most_representative_geometry(gdf)
            if kind == 'point' or kind == 'line' or kind == 'polygon':
                gdf = GeoTools._separate_geometries(gdf)[kind.capitalize()]
        return gdf

    # Returns UTM EPSG code based on a place
    def get_utm_epsg_from_place(place):
        ox.settings.timeout = GeoTools.timeout
        lat, lng = ox.geocoder.geocode(place)
        return GeoTools.get_utm_epsg_from_lat_lng(lat, lng)


    # Returns UTM EPSG code based on a lat lng coordinate
    def get_utm_epsg_from_lat_lng(lat, lng):

        if lat<0:
            south = True
        else:
            south = False
        utm_x, utm_y, zone, type = utm.from_latlon(lat,lng)
        crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': south})
        return int(crs.to_authority()[1])


    # Returns a grid of points in the geometry (GDF) specified
    def grid_points_based_on_polygon(gdf, gap, utm):
        area = gdf.copy().to_crs(utm)
        minx, miny, maxx, maxy = area.unary_union.bounds

        point_list = []
        x = minx
        y = miny
        while x <= maxx:
            while y <= maxy:
                p_aux = Point(x, y)
                point_list.append(p_aux)
                y += gap
            y = miny
            x += gap
        
        points = {'geometry': point_list}
        grid_points = gpd.GeoDataFrame(points, crs = utm)
        grid_points = grid_points.clip(area)
        grid_points = grid_points.to_crs(gdf.crs)
        grid_points = grid_points.reset_index(drop=True)
        grid_points['consulted'] = 0
        return grid_points

    def create_gdf_from_panoids(panoids):
        df = pd.DataFrame(panoids)
        df['geometry'] = df.apply(lambda row: Point(row['lng'], row['lat']), axis = 1)
        gdf = gpd.GeoDataFrame(df, crs = 4326)
        return gdf


    def get_angle_of_line(line, point, dist, buffer = 2):
        panoid_buffer = point.buffer(dist + buffer)
        short_line = line.intersection(panoid_buffer)

        if short_line.is_empty:
            return None
        
        if short_line.geom_type == 'MultiLineString':
            short_line = short_line[0]

        p1 = Point(*short_line.coords[0])
        p2 = Point(*short_line.coords[-1])

        if (p2.x - p1.x) == 0:
            return 90
        else:
            m = (p2.y - p1.y)/(p2.x - p1.x)
            angle_rad = math.atan(m)
            angle = round(angle_rad*180/math.pi,3)
            return -angle

    ## EXTRA TOOLS

    # Returns the same GeoDataFrame of the input with columns diff of NaN
    # and with simplified headers
    # output: GeoDataFrame
    def _simplify_gdf_columns(gdf):
        gdf = gdf.dropna(axis = 1, how = 'all')
        keys = ['name:']
        for k in keys:
            gdf = gdf.loc[:,gdf.columns.drop(list(gdf.filter(regex=k)))]
        
        gdf = gdf.reset_index(drop = True)
        gdf = GeoTools._remove_columns_with_lists(gdf)
        return gdf


    # Returns the same GeoDataFrame of the input without values as lists
    # output: GeoDataFrame
    def _remove_columns_with_lists(gdf):
        for col in gdf.columns:
            if any(isinstance(val, list) for val in gdf[col]):
                gdf[col] = gdf[col].apply(lambda x: str(x))
        return gdf


    # Separates the GeoDataFrame by Geometries types
    def _separate_geometries(gdf):
        gdf['geom_type'] = gdf['geometry'].apply(lambda x: x.geom_type)
        geoms = ['Point', 'LineString', 'Polygon']
        for i, g in enumerate(geoms):
            rule = (gdf['geom_type'] == g) | (gdf['geom_type'] == f'Multi{g}')
            one_geo_gdf = gdf.copy().loc[rule].drop(columns=['geom_type']) 
            geoms[i] = one_geo_gdf if len(one_geo_gdf) > 0 else None
        return {'Point': geoms[0], 'Line': geoms[1], 'Polygon': geoms[2]}


    # Return a gdf with the most repeated geoemtry
    def _most_representative_geometry(gdf):
        gdfs = GeoTools._separate_geometries(gdf)
        fig = ''
        n = 0
        for g in gdfs:
            if gdfs[g] is None:
                continue
            if len(gdfs[g]) >= n:
                n = len(gdfs[g])
                fig = g
        return gdfs[fig]
    
    # Send a HTTP GET request to the Nominatim API and return JSON response.
    def nominatim_request(params, request_type="search", pause=1, error_pause=60):

        if request_type not in {"search", "reverse", "lookup"}:  # pragma: no cover
            raise ValueError('Nominatim request_type must be "search", "reverse", or "lookup"')

        # resolve url to same IP even if there is server round-robin redirecting
        ox.downloader._config_dns(ox.settings.nominatim_endpoint.rstrip("/"))

        # prepare Nominatim API URL and see if request already exists in cache
        url = ox.settings.nominatim_endpoint.rstrip("/") + "/" + request_type

        prepared_url = requests.Request("GET", url, params=params).prepare().url
        cached_response_json = ox.downloader._retrieve_from_cache(prepared_url)

        if ox.settings.nominatim_key:
            params["key"] = ox.settings.nominatim_key

        if cached_response_json is not None:
            # found response in the cache, return it instead of calling server
            return cached_response_json

        else:
            # if this URL is not already in the cache, pause, then request it
            ox.utils.log(f"Pausing {pause} seconds before making HTTP GET request")
            time.sleep(pause)

            # transmit the HTTP GET request
            ox.utils.log(f"Get {prepared_url} with timeout={ox.settings.timeout}")
            headers = ox.downloader._get_http_headers()
            response = requests.get(
                url,
                params=params,
                timeout=ox.settings.timeout,
                headers=headers,
                **ox.settings.requests_kwargs,
            )
            sc = response.status_code

            # log the response size and domain
            size_kb = len(response.content) / 1000
            domain = re.findall(r"(?s)//(.*?)/", url)[0]
            ox.utils.log(f"Downloaded {size_kb:,.1f}kB from {domain}")

            try:
                response_json = response.json()

            except Exception:  # pragma: no cover
                if sc in {429, 504}:
                    # 429 is 'too many requests' and 504 is 'gateway timeout' from
                    # server overload: handle these by pausing then recursively
                    # re-trying until we get a valid response from the server
                    ox.utils.log(f"{domain} returned {sc}: retry in {error_pause} secs", level=logging.WARNING)
                    time.sleep(error_pause)
                    response_json = GeoTools.nominatim_request(params, request_type, pause, error_pause)

                else:
                    # else, this was an unhandled status code, throw an exception
                    ox.utils.log(f"{domain} returned {sc}", level=logging.ERROR)
                    raise Exception(f"Server returned:\n{response} {response.reason}\n{response.text}")

            ox.downloader._save_to_cache(prepared_url, response_json, sc)
            return response_json
    
