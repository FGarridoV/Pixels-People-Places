import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import Point, LineString


def open_panoids(panoids_filepath: str) -> gpd.GeoDataFrame:
    panoids = gpd.read_file(panoids_filepath, columns=['panoid', 'year', 'im_front', 'im_back'])
    panoids = panoids.to_crs('EPSG:28992')

    for index, panoid in panoids.iterrows():
        point = panoid['geometry']
        panoids.loc[index, 'geometry'] = Point(int(round(point.x)), int(round(point.y)))

    print(f"{len(panoids)} panoids present in given file")
    return panoids


def open_network(network_filepath: str, id_column: str, only_id_column: bool=True,
                 create_unique_ids: bool=False) -> gpd.GeoDataFrame:
    if only_id_column:
        network = gpd.read_file(network_filepath, columns=[id_column])
    else:
        network = gpd.read_file(network_filepath)

    network[id_column] = network[id_column].astype('str')
    network = network.to_crs('EPSG:28992')

    if create_unique_ids:
        print("Unique IDs will be writen to new column 'ID'. If column already exists it will be overwritten")
        network['ID'] = [None] * len(network)
        for index, section in network.iterrows():
            network.loc[index, 'ID'] = f"{section[id_column]}_{index}"
        id_column = 'ID'
    else:
        network['ID'] = network[id_column]

    print(f"{len(network)} sections in given network file")
    return network


def remove_small_sections(network: gpd.GeoDataFrame, min_length: int) -> gpd.GeoDataFrame:
    network = network.copy()

    original_size = len(network)
    print(f"Original network contains {original_size} sections")
    print(f"Removing all sections with a length smaller than {min_length} meters")
    network = network[[section.length >= min_length for section in network['geometry'].values]]
    network = network.reset_index(drop=True)
    new_size = len(network)
    print(f"Removed {original_size-new_size} sections, new network contains {new_size} sections")

    return network




def simplify(network: gpd.GeoDataFrame, min_length: int) -> gpd.GeoDataFrame:
    network = network.copy()

    # TODO: STILL DOES NOT WORK!! Check with other network? maybe remove int if that one is better
    # Maybe make geometries around points, find all within, and make index for those.

    network['FIRST'] = [Point(int(section.coords[0][0]), int(section.coords[0][1])) for section in network['geometry'].values]
    network['LAST'] = [Point(int(section.coords[-1][0]), int(section.coords[-1][1])) for section in network['geometry'].values]
    network['LENGTH'] = [section.length for section in network['geometry'].values]
    network['FIRST_UPDATED'] = [False] * len(network)
    network['LAST_UPDATED'] = [False] * len(network)

    to_delete = network[network['LENGTH'] < min_length].index
    print(len(to_delete))

    for index, section in network.iterrows():
        if index in to_delete:
            print(index)
            first = section['FIRST']
            last = section['LAST']
            middle = Point(int((first.x+last.x)/2), int((first.y+last.y)/2))

            network.loc[network['FIRST'].isin([first, last]), ['FIRST', 'FIRST_UPDATED']] = [middle, True]
            network.loc[network['LAST'].isin([first, last]), ['LAST', 'LAST_UPDATED']] = [middle, True]

    network = network[~network.index.isin(to_delete)].copy()
    for index, section in network.iterrows():
        new_line = list(section['geometry'].coords)

        if section['FIRST_UPDATED']:
            new_line.insert(0, (section['FIRST'].x, section['FIRST'].y))
        if section['LAST_UPDATED']:
            new_line.append((section['LAST'].x, section['LAST'].y))

        network.loc[index, 'geometry'] = LineString(new_line)

    network = network.drop(columns=['FIRST', 'LAST', 'LENGTH', 'FIRST_UPDATED', 'LAST_UPDATED'])
    network = network.reset_index(drop=True)
    return network


def match_panoids_to_sections(panoids: gpd.GeoDataFrame, network: gpd.GeoDataFrame, maximum_distance: int) -> dict[str, list[str]]:
    network_panoids =  {key: [] for key in network['ID'].values}

    count = 0
    for _, panoid in panoids.iterrows():
        panoid_point = panoid['geometry']

        network_i = network.cx[panoid_point.x-maximum_distance:panoid_point.x+maximum_distance,
                               panoid_point.y-maximum_distance:panoid_point.y+maximum_distance].copy()
        network_i['distance'] = network_i.distance(panoid_point)

        network_i = network_i[network_i['distance'] <= maximum_distance].copy()
        if len(network_i) > 0:
            network_i.sort_values(by=['distance'])
            network_panoids[network_i['ID'].values[0]].append(panoid['panoid'])

            if len(network_i) >= 2 and network_i['distance'].values[0] == network_i['distance'].values[1]:
                network_panoids[network_i['ID'].values[1]].append(panoid['panoid'])

            count += 1
    print(f"Matched {count} out of {len(panoids)} panoids to network sections")

    print(f"Only network sections with assigned panoids will keep their entry in the assignment dictionary")
    to_delete = []
    for key, value in network_panoids.items():
        if value == []:
            to_delete.append(key)
    for key_to_delete in to_delete:
        del network_panoids[key_to_delete]

    print(f"{len(network_panoids)} out of {len(network)} network sections have had panoids assigned")

    return network_panoids


def assign_images_to_sections(network_panoids: dict[str, list[str]], panoids: gpd.GeoDataFrame,
                              project_name: str) -> dict[str, list[str]]:
    network_images =  {key: [] for key in network_panoids.keys()}

    print("Gathering all assigned images for which image features are present")
    for section_id, panoid_ids in network_panoids.items():
        for panoid_id in panoid_ids:
            panoid = panoids[panoids['panoid'] == panoid_id]
            for image in [panoid['im_back'].values[0], panoid['im_front'].values[0]]:
                if os.path.exists(f"Intermediate Data/{project_name}/image_features/{image}.npy"):
                    network_images[section_id].append(image)
                else:
                    print(f"WARNING: features of '{image}' not found in project features folder, will be skipped")
    print("Gathered all assigned images")    

    print("Checking if missing images resulted in sections with panoid assignements but wihtout image assignments")
    to_delete = []
    for section_id, images in network_images.items():
        if images == []:
            to_delete.append(section_id)

    print(f"{len(to_delete)} section(s) with empty image assignments due to missing images, empty assignments will be deleted")
    for delete_id in to_delete:
        del network_images[delete_id]

    return network_images


def plot_image_assignments(network: gpd.GeoDataFrame, image_assignments: dict[str, list[str]],
                           save: bool=False) -> None:
    network = network.copy()
    network['color'] = ['g' if section_id in image_assignments else 'r' for section_id in network['ID']]

    fig, ax = plt.subplots(figsize=(8,8))
    network.plot(ax=ax, color=network['color'])

    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        fig.savefig(r'Images\Network Assignment Coverage.png', bbox_inches='tight', dpi=300)

    ax.set_title("Network sections image assignments\n(with assignments in green and without in red)")

    plt.show()


def plot_data(network: gpd.GeoDataFrame, panoids: gpd.GeoDataFrame, save: bool=True) -> None:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,8), sharey=True)

    network.plot(ax=ax1, linewidth=0.9)
    #ax1.set_xlabel('Rijksdriehoek (RD_NEW) X coordinate')
    #ax1.set_xlabel('Rijksdriehoek (RD_NEW) Y coordinate')

    panoids.plot(ax=ax2, markersize=0.05)
    #ax2.set_xlabel('Rijksdriehoek (RD_NEW) X coordinate')

    for ax in [ax1, ax2]:
        ax.set_ylim([442200, 450000])
        ax.set_xlim([81500, 87750])

    fig.tight_layout()
    fig.text(0.5, 0.17, 'RD New X coordinate', ha='center')
    fig.text(-0.01, 0.5, 'RD New Y coordinate', va='center', rotation='vertical')

    if save:
        fig.savefig(r'Images\Network_and_Panoids.png', bbox_inches='tight', dpi=300)
    
    fig.show()
