import pandas as pd
import geopandas as gpd

def aggregate_detections_on_cells(detections, cells, classes, min_year, max_year, criteria = 'avg'):

    # Filtering detections by year
    cond1 = (detections['year'] >= min_year)
    cond2 = (detections['year'] <= max_year)
    cond3 = (detections[classes[0] + '_det'] != -1)# Remove -1 (images not found)
    agg_detections = detections.loc[cond1 & cond2 & cond3]

    # Adding which cell belongs to each sample
    class_cols = [c + '_det' for c in classes]
    columns = ['cell'] + class_cols
    agg_detections = gpd.sjoin(agg_detections, cells[['cell', 'geometry']], how = 'left')[columns]

    # Aggregating information based on criteria
    if criteria == 'avg':
        agg_detections = agg_detections.groupby(by = 'cell').mean()
    elif criteria == 'sum':
        agg_detections = agg_detections.groupby(by = 'cell').sum()

    # Adding aggregated data on cells
    gdf_grid = pd.merge(cells, agg_detections, how = 'left', on = 'cell')
    gdf_grid['has_data'] = gdf_grid[class_cols[0]].apply(lambda x: 0 if pd.isna(x) else 1)
    gdf_grid = gdf_grid.fillna(0)
    return gdf_grid


def aggregate_network_info_on_cells(edges, nodes, cells):

    # Adding belonging cell to edges
    edges_info = edges.copy()
    edges_info = edges_info.loc[edges_info.geometry.geometry.type == 'LineString']
    agg_edges = edges_info.overlay(cells, how='intersection')
    agg_edges['m_street'] = agg_edges.length
    agg_edges = agg_edges[['cell', 'm_street']]

    # Adding belonging cell to nodes
    nodes_info = nodes.copy()
    nodes_info['n_nodes'] = 1
    agg_nodes = gpd.sjoin(nodes_info, cells[['cell', 'geometry']], how = 'left')[['cell', 'n_nodes']]

    # Aggregating information based on criteria
    agg_edges = agg_edges.groupby(by = 'cell').sum().reset_index()
    agg_nodes = agg_nodes.groupby(by = 'cell').sum().reset_index()

    # Adding aggregated data on cells
    gdf_grid = pd.merge(cells, agg_edges[['cell', 'm_street']], how = 'left', on = 'cell')
    gdf_grid = pd.merge(gdf_grid, agg_nodes[['cell', 'n_nodes']], how = 'left', on = 'cell')
    gdf_grid = gdf_grid.fillna(0)
    return gdf_grid

# Amenities
def aggregate_point_info_on_cells(points, cells, category_col):
    points_info = points.copy()
    classes = points[category_col].drop_duplicates().to_list()

    points_info[classes] = [0]*len(classes)
    def assign_row(gdf, row):
        gdf.at[row.name, row[category_col]] = 1
    points_info.apply(lambda row: assign_row(points_info, row), axis = 1)

    agg_points = gpd.sjoin(points_info, cells[['cell', 'geometry']], how = 'left')[['cell'] + classes]
    agg_points = agg_points.groupby(by = 'cell').sum().reset_index()

    class_cols = {c:c + '_amn' for c in classes}
    agg_points.rename(columns = class_cols, inplace = True)

    # Adding aggregated data on cells
    gdf_grid = pd.merge(cells, agg_points, how = 'left', on = 'cell')
    gdf_grid = gdf_grid.fillna(0)
    return gdf_grid

def aggregate_polygon_info_on_cells(polygons, cells, category_col):
    polygons_info = polygons.copy()
    polygons_info = polygons_info.dissolve('landuse').reset_index()
    classes = polygons_info[category_col].drop_duplicates().to_list()
    agg_polygons = polygons_info.overlay(cells, how='intersection')
    agg_polygons['surface'] = agg_polygons.area
    agg_polygons[classes] = [0]*len(classes)

    def assign_row(gdf, row):
        gdf.at[row.name, row[category_col]] = row['surface']
    agg_polygons.apply(lambda row: assign_row(agg_polygons, row), axis = 1)
    agg_polygons = agg_polygons.loc[:,['cell'] + classes]

    agg_polygons = agg_polygons.groupby(by = 'cell').sum().reset_index()

    class_cols = {c:c + '_lu' for c in classes}
    agg_polygons.rename(columns = class_cols, inplace = True)

    # Adding aggregated data on cells
    gdf_grid = pd.merge(cells, agg_polygons, how = 'left', on = 'cell')
    gdf_grid = gdf_grid.fillna(0)
    return gdf_grid

