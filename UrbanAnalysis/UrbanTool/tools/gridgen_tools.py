import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Polygon
from math import cos, sin, pi


def create_grid_on_gdf(gdf, shape = 'rectangle', width = 200, height = 200, side = 200, crop = 'hex'):
    # crop: 'full', 'hex', 'non'
    xmin, ymin, xmax, ymax = gdf.bounds.iloc[0].to_list()
    epsg = gdf.crs

    grid = create_grid(xmin, ymin, xmax, ymax, epsg, shape = shape, width = width, height = height, side = side)

    if crop == 'full':
        grid = gpd.clip(grid, gdf)
        grid = grid.reset_index(drop = True)
        grid['cell'] = grid.index
    elif crop == 'hex':
        gdf['union'] = 0
        gdf = gdf.dissolve(by = 'union')
        grid = gpd.sjoin(grid, gdf, how = 'inner')
        grid = grid.loc[:,['cell', 'area', 'geometry']]
        grid = grid.reset_index(drop=True)
        grid['cell'] = grid.index
    return grid
    
    


def create_grid(xmin, ymin, xmax, ymax, epsg, shape = 'rectangle', width = 200, height = 200, side = 200):

    if shape == 'rectangle':
        cells = _rectangle_tessellation(xmin, xmax, ymin, ymax, width, height, epsg)

    elif shape == 'hexagon':
        cells = _hexagon_tessellation(xmin, xmax, ymin, ymax, side, epsg)

    cells['cell'] = range(1, cells.shape[0] +1)
    return cells

# This method returns a grid tessellation shape based on:
## xmin: left x border to fulfill with grid (UTM coordinate)
## xmax: right x border to fulfill with grid (UTM coordinate)
## ymin: bottom y border to fulfill with grid (UTM coordinate)
## ymax: top y border to fulfill with grid (UTM coordinate)
## width: width of specific cell in grid teselation (meters)
## heigth: heigth of specific cell in grid teselation (meters)
## epsg: projection used in the input coordinates (UTM)
def _rectangle_tessellation(xmin, xmax, ymin, ymax, width, height, epsg):
    w = width
    h = height
    
    rows = int(np.ceil((ymax-ymin) / height))
    cols = int(np.ceil((xmax-xmin) / width))
    
    xCol = xmin
    yCol = ymax
    
    cells = []
    polygons = []
    c = 1
    for _ in range(cols):
        x = xCol
        y = yCol
        xCol += w
        for _ in range(rows):
            cells.append(c)
            polygons.append(Polygon([(x,y), (x+w,y), (x+w,y-h), (x,y-h)])) 
            y -= h
            c+=1

    tess = gpd.GeoDataFrame({'cell': cells, 'geometry':polygons})
    tess.crs = epsg
    tess['area'] = tess.area
    columns = tess.columns.to_list()
    columns.remove('geometry')
    columns.append('geometry') 
    tess = tess[columns]
    return tess

# This method returns a hexagon teselation shape based on:
## xmin: left x border to fulfill with grid (UTM coordinate)
## xmax: right x border to fulfill with grid (UTM coordinate)
## ymin: bottom y border to fulfill with grid (UTM coordinate)
## ymax: top y border to fulfill with grid (UTM coordinate)
## side: hexagon radious or side (meters)
## epsg: projection used in the input coordinates (UTM)
def _hexagon_tessellation(xmin, xmax, ymin, ymax, side, epsg):
    s = side
    a = s*cos(pi/6)
    b = s*sin(pi/6)
    
    rows = int(np.ceil((ymax-ymin) / (2*a)))
    cols = int(np.ceil((xmax-xmin) / (b+s)))
    
    xEvenCol = xmin
    yEvenCol = ymax
    xOddCol = xmin + s + b
    yOddCol = ymax + a
    
    cells = []
    polygons = []
    
    c = 1
    for i in range(cols+1):
        if i%2 == 0:
            x = xEvenCol
            y = yEvenCol
            xEvenCol += 2*(b+s) 
        else:
            x = xOddCol
            y = yOddCol
            xOddCol += 2*(b+s)
        for _ in range(rows+1):
            cells.append(c)
            polygons.append(Polygon([(x, y), (x+s,y), (x+s+b,y-a), (x+s,y-2*a),(x,y-2*a),(x-b,y-a)])) 
            y -= 2*a
            c+=1 
    tess = gpd.GeoDataFrame({'cell': cells, 'geometry':polygons})
    tess.crs = epsg
    tess['area'] = tess.area
    columns = tess.columns.to_list()
    columns.remove('geometry')
    columns.append('geometry') 
    tess = tess[columns]
    return tess