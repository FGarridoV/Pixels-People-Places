import rioxarray
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from matplotlib import pyplot as plt
from geocube.vector import vectorize
plt.rcParams.update({'font.size': 16})


def plot_amount_of_images_per_cluster(network: gpd.GeoDataFrame, image_assignments: dict[str, str],
                                      cluster_colors: dict[str, str]) -> None:
    clustering_cols = ['cluster_km', 'cluster_hr', 'cluster_gm']
    data = {}
    for cluster in network['cluster_km'].unique():
        if cluster != None:
            data[cluster] = []

    for clustering_col in clustering_cols:
        for cluster in network[clustering_col].unique():
            if cluster != None:
                matches = network[network[clustering_col] == cluster]

                count = sum(len(image_assignments[match['ID']]) for _, match in matches.iterrows())
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())
                data[cluster].append(count / total_length)

    # Sort data by descending order of kmeans cluster figures
    kmeans_data = [data[cluster][0] for cluster in data.keys()]
    sorted_keys = [x for _, x in sorted(zip(kmeans_data, data.keys()), reverse=True)]
    sorted_data = {key: data[key] for key in sorted_keys}


    x = np.arange(len(clustering_cols))
    width = (1-0.2) / len(list(sorted_data.keys())) 
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(12, 6))


    for cluster, values in sorted_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=cluster, color=cluster_colors[cluster], zorder=2)
        multiplier += 1

    # ax.set_title("Pictures per length per cluster per clustering type")
    ax.set_ylabel("Number of pictures per meter of road [#/m]")
    ax.set_xlabel("Clustering method")
    ax.set_xticks(x + width * ((len(list(sorted_data.keys()))/2) - 0.5))
    ax.set_xticklabels(['K-Means', 'Hierarchical', 'Gaussian Mixture'])
    ax.grid(which='major', axis='y', zorder=1)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', title='Cluster')
    fig.tight_layout()
    fig.savefig(r'Images\Picture per Length.png')

    plt.show()


def plot_osm_road_types_comparison(which_set: str, network: gpd.GeoDataFrame) -> None:
    if which_set not in ['hierarchical', 'all']:
        raise ValueError("which_set can only be 'hierarchical' or 'all'")

    if which_set == 'hierarchical':
        clustering_cols = ['cluster_hr']
        width = 0.5

        clusters = ['City centric', 'Residential A', 'Residential B', 'Greenery',
                    'Arterial', 'Motorway']
        clusters = list(reversed(clusters))
        road_types = ['pedestrian', 'living_street', 'residential', 'tertiary', 'secondary', 'primary',
                    'trunk', 'motorway_link', 'motorway', 'service', 'unclassified']

        colors = plt.get_cmap("YlOrRd", 9)
        colors = colors(np.arange(0, 1, 1/(len(road_types)-2)))

        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        for ax, clustering_col in zip([axs], clustering_cols):
            data = {road_type: [0] * len(clusters) for road_type in road_types}
            for i, cluster in enumerate(clusters):
                matches = network[network[clustering_col] == cluster]
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())

                for road_type in road_types:        
                    matches_2 = matches[matches['highway'] == str(road_type)]

                    if len(matches_2) != 0:
                        length = sum(match['geometry'].length for _, match in matches_2.iterrows())
                        data[road_type][i] = length / total_length

            # Plot bars
            bottom = np.zeros(len(clusters))
            for i, (road_type, values) in enumerate(data.items()):
                if i == 9:
                    p = ax.barh(clusters, values, width, label=road_type, left=bottom, zorder=2, color='#D3D3D3')
                elif i == 10:
                    p = ax.barh(clusters, values, width, label=road_type, left=bottom, zorder=2, color='#A9A9A9')
                else:
                    p = ax.barh(clusters, values, width, label=road_type, left=bottom, zorder=2, color=colors[i])
                bottom += values

                ax.legend(bbox_to_anchor=(1.109, 1.0), loc='upper left', title='Road type')
            ax.grid(which='major', axis='x', zorder=1)
            ax.set_ylabel('Cluster')
            ax.set_xlabel('Fraction')
            ax.set_xlim(0, 1)

        fig.tight_layout()
        fig.savefig("Images/Road Types Hierarchical.png", bbox_inches='tight', dpi=300)
        plt.show()

    else:  # which_set == 'all':
        clustering_cols = ['cluster_km', 'cluster_hr', 'cluster_gm']
        titles = ['K-Means', 'Hierarchical', 'Gaussian Mixture']
        width = 0.5

        clusters = ['City centric', 'Residential A', 'Residential B', 'Greenery',
                    'Arterial', 'Motorway']
        clusters = list(reversed(clusters))
        road_types = ['pedestrian', 'living_street', 'residential', 'tertiary', 'secondary', 'primary',
                      'trunk', 'motorway_link', 'motorway', 'service', 'unclassified']

        colors = plt.get_cmap("YlOrRd", 9)
        colors = colors(np.arange(0, 1, 1/(len(road_types)-2)))

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, clustering_col, title in zip(axs, clustering_cols, titles):
            data = {road_type: [0] * len(clusters) for road_type in road_types}
            for i, cluster in enumerate(clusters):
                matches = network[network[clustering_col] == cluster]
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())

                for road_type in road_types:        
                    matches_2 = matches[matches['highway'] == str(road_type)]

                    if len(matches_2) != 0:
                        length = sum(match['geometry'].length for _, match in matches_2.iterrows())
                        data[road_type][i] = length / total_length

            # Plot bars
            bottom = np.zeros(len(clusters))
            for i, (road_type, values) in enumerate(data.items()):
                if i == 9:
                    p = ax.barh(clusters, values, width, label=road_type, left=bottom, zorder=2, color='#D3D3D3')
                elif i == 10:
                    p = ax.barh(clusters, values, width, label=road_type, left=bottom, zorder=2, color='#A9A9A9')
                else:
                    p = ax.barh(clusters, values, width, label=road_type, left=bottom, zorder=2, color=colors[i])
                bottom += values

            if ax == axs[-1]:
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', title='Road type')
            if ax == axs[1]:
                ax.set_xlabel('Fraction')
            if ax == axs[0]:
                ax.set_ylabel('Cluster')
            if ax != axs[0]:
                ax.set_yticklabels([])
            ax.grid(which='major', axis='x', zorder=1)

            ax.set_xlim(0, 1)
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig("Images/Road Types.png", bbox_inches='tight', dpi=300)
        plt.show()


def plot_osm_surface_type_comparison(which_set: str, network: gpd.GeoDataFrame) -> None:
    if which_set not in ['hierarchical', 'all']:
        raise ValueError("which_set can only be 'hierarchical' or 'all'")
    
    if which_set == 'hierarchical':
        clustering_cols = ['cluster_hr']
        width = 0.5

        clusters = list(network['cluster_km'].unique())
        clusters.remove(None)
        surfaces = ['asphalt', 'paving_stones', 'other']
        other_surfaces = list(network['surface'].unique())
        other_surfaces.remove(None)
        other_surfaces.remove('asphalt')
        other_surfaces.remove('paving_stones')

        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        for ax, clustering_col in zip([axs], clustering_cols):
            data = {speed: [0] * len(clusters) for speed in surfaces}
            for i, cluster in enumerate(clusters):
                matches = network[network[clustering_col] == cluster]
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())

                for surface in surfaces:
                    if surface == 'other':
                        matches_2 = matches[matches['surface'].isin(other_surfaces)]
                    else:
                        matches_2 = matches[matches['surface'] == str(surface)]

                    if len(matches_2) != 0:
                        length = sum(match['geometry'].length for _, match in matches_2.iterrows())
                        data[surface][i] = length / total_length

            # Sort by share of asphalt per cluster from the kmeans clustering ascending
            sort_by = data['asphalt']
            clusters_sorted = [x for _, x in sorted(zip(data['asphalt'],clusters), reverse=True)]

            data_sorted = data.copy()
            for surface, values in data.items():
                data_sorted[surface] = [x for _, x in sorted(zip(sort_by,values), reverse=True)]

            # Plot bars
            bottom = np.zeros(len(clusters_sorted))
            for surface, values in data_sorted.items():
                if surface == 'other':
                    p = ax.barh(clusters_sorted, values, width, label=surface, left=bottom, zorder=2, color='#A9A9A9')
                else:
                    p = ax.barh(clusters_sorted, values, width, label=surface, left=bottom, zorder=2)
                bottom += values


            ax.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left', title='Surface type')
            ax.grid(which='major', axis='x', zorder=1)
            ax.set_ylabel('Cluster')
            ax.set_xlabel('Fraction')
            ax.set_xlim(0, 1)

        fig.tight_layout()
        fig.savefig("Images/Surface Types Hierarchical.png", bbox_inches='tight', dpi=300)
        plt.show()

    else:  # which_set == 'all':
        clustering_cols = ['cluster_km', 'cluster_hr', 'cluster_gm']
        titles = ['K-Means', 'Hierarchical', 'Gaussian Mixture']
        width = 0.5

        clusters = list(network['cluster_km'].unique())
        clusters.remove(None)
        surfaces = ['asphalt', 'paving_stones', 'other']
        other_surfaces = list(network['surface'].unique())
        other_surfaces.remove(None)
        other_surfaces.remove('asphalt')
        other_surfaces.remove('paving_stones')

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, clustering_col, title in zip(axs, clustering_cols, titles):
            data = {speed: [0] * len(clusters) for speed in surfaces}
            for i, cluster in enumerate(clusters):
                matches = network[network[clustering_col] == cluster]
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())

                for surface in surfaces:        
                    if surface == 'other':
                        matches_2 = matches[matches['surface'].isin(other_surfaces)]
                    else:
                        matches_2 = matches[matches['surface'] == str(surface)]

                    if len(matches_2) != 0:
                        length = sum(match['geometry'].length for _, match in matches_2.iterrows())
                        data[surface][i] = length / total_length

            # Sort by share of asphalt per cluster from the kmeans clustering ascending
            if ax == axs[0]:
                sort_by = data['asphalt']
                clusters_sorted = [x for _, x in sorted(zip(data['asphalt'],clusters), reverse=True)]

            data_sorted = data.copy()
            for surface, values in data.items():
                data_sorted[surface] = [x for _, x in sorted(zip(sort_by,values), reverse=True)]

            # Plot bars
            bottom = np.zeros(len(clusters_sorted))
            for surface, values in data_sorted.items():
                if surface == 'other':
                    p = ax.barh(clusters_sorted, values, width, label=surface, left=bottom, zorder=2, color='#A9A9A9')
                else:
                    p = ax.barh(clusters_sorted, values, width, label=surface, left=bottom, zorder=2)
                bottom += values

            if ax == axs[-1]:
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', title='Surface type')
            if ax == axs[1]:
                ax.set_xlabel('Fraction')
            if ax == axs[0]:
                ax.set_ylabel('Cluster')
            if ax != axs[0]:
                ax.set_yticklabels([])
            ax.grid(which='major', axis='x', zorder=1)
            ax.set_xlim(0, 1)
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig("Images/Surface Types.png", bbox_inches='tight', dpi=300)
        plt.show()


def plot_osm_speed_limits_comparison(which_set: str, network: gpd.GeoDataFrame) -> None:
    if which_set not in ['hierarchical', 'all']:
        raise ValueError("which_set can only be 'hierarchical' or 'all'")

    if which_set == 'hierarchical':
        clustering_cols = ['cluster_hr']

        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        clusters = list(network['cluster_km'].unique())
        clusters.remove(None)
        speeds = list(network['maxspeed'].unique())
        speeds.remove(None)
        speeds = [int(speed) for speed in speeds]
        speeds.sort()

        my_cmap = plt.get_cmap("RdYlGn_r")
        rescale = lambda speed: (speed - np.min(speeds)) / (np.max(speeds) - np.min(speeds))
        width = 0.5

        for ax, clustering_col in zip([axs], clustering_cols):

            data = {speed: [0] * len(clusters) for speed in speeds}
            for i, cluster in enumerate(clusters):
                matches = network[network[clustering_col] == cluster]
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())

                for speed in speeds:        
                    matches_2 = matches[matches['maxspeed'] == str(speed)]

                    if len(matches_2) != 0:
                        length = sum(match['geometry'].length for _, match in matches_2.iterrows())
                        data[speed][i] = length / total_length

            # Soort by largest share before 50 km/h first from the kmeans clustering
            sort_by = np.array(data[5]) + np.array(data[15]) + np.array(data[30])
            clusters_sorted = [x for _, x in sorted(zip(sort_by,clusters), reverse=False)]

            data_sorted = data.copy()
            for speed, values in data.items():
                data_sorted[speed] = [x for _, x in sorted(zip(sort_by, values), reverse=False)]

            # Make transition between 30 an 50 km/h the centre
            bottom = np.zeros(len(clusters_sorted))
            for i, cluster in enumerate(clusters_sorted):
                bottom[i] = -np.sum([data_sorted[speed][i] for speed in speeds[:3]])

            # Plot bars
            for speed, values in data_sorted.items():
                p = ax.barh(clusters_sorted, values, width, label=speed, left=bottom, zorder=2, color=my_cmap(rescale(speed)))
                bottom += values

            # Pretty up graphs
            ax.legend(bbox_to_anchor=(1.08, 1.0), loc='upper left', title='Speed limits (km/h)')
            ax.set_xlabel('Fraction')
            ax.set_ylabel('Cluster')
            ax.grid(which='major', axis='x', zorder=1)
            ax.set_xlim([-1, 1])
            ax.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(np.abs(ax.get_xticks()))

        fig.tight_layout()
        fig.savefig("Images/Speed Limits Hierarchical.png", bbox_inches='tight', dpi=300)
        plt.show()

    else:  # which_set == 'all':
        clustering_cols = ['cluster_km', 'cluster_hr', 'cluster_gm']
        titles = ['K-Means', 'Hierarchical', 'Gaussian Mixture']

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        clusters = list(network['cluster_km'].unique())
        clusters.remove(None)
        speeds = list(network['maxspeed'].unique())
        speeds.remove(None)
        speeds = [int(speed) for speed in speeds]
        speeds.sort()

        my_cmap = plt.get_cmap("RdYlGn_r")
        rescale = lambda speed: (speed - np.min(speeds)) / (np.max(speeds) - np.min(speeds))
        width = 0.5

        for ax, clustering_col, title in zip(axs, clustering_cols, titles):

            data = {speed: [0] * len(clusters) for speed in speeds}
            for i, cluster in enumerate(clusters):
                matches = network[network[clustering_col] == cluster]
                total_length = sum(match['geometry'].length for _, match in matches.iterrows())

                for speed in speeds:        
                    matches_2 = matches[matches['maxspeed'] == str(speed)]

                    if len(matches_2) != 0:
                        length = sum(match['geometry'].length for _, match in matches_2.iterrows())
                        data[speed][i] = length / total_length

            # Soort by largest share before 50 km/h first from the kmeans clustering
            if ax == axs[0]:
                sort_by = np.array(data[5]) + np.array(data[15]) + np.array(data[30])
                clusters_sorted = [x for _, x in sorted(zip(sort_by,clusters), reverse=False)]

            data_sorted = data.copy()
            for speed, values in data.items():
                data_sorted[speed] = [x for _, x in sorted(zip(sort_by, values), reverse=False)]

            # Make transition between 30 an 50 km/h the centre
            bottom = np.zeros(len(clusters_sorted))
            for i, cluster in enumerate(clusters_sorted):
                bottom[i] = -np.sum([data_sorted[speed][i] for speed in speeds[:3]])

            # Plot bars
            for speed, values in data_sorted.items():
                p = ax.barh(clusters_sorted, values, width, label=speed, left=bottom, zorder=2, color=my_cmap(rescale(speed)))
                bottom += values

            # Pretty up graphs
            if ax == axs[-1]:
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left', title='Speed limits (km/h)')
            if ax == axs[1]:
                ax.set_xlabel('Fraction')
            if ax == axs[0]:
                ax.set_ylabel('Cluster')
            if ax != axs[0]:
                ax.set_yticklabels([])
            ax.grid(which='major', axis='x', zorder=1)
            ax.set_xlim([-1, 1])
            ax.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(np.abs(ax.get_xticks()))
            ax.set_title(title)

        fig.tight_layout()
        fig.savefig("Images/Speed Limits.png", bbox_inches='tight', dpi=300)
        plt.show()


def open_groen_data(groen_file: str, network: gpd.GeoDataFrame, plot: bool=True):
    data = rioxarray.open_rasterio(r'Groen/rivm_20231221_groenkaart_10m_2022.tif')
    data.name = "myData"
    # gdf = vectorize(data)

    bounds = network.total_bounds

    data = data.rio.clip_box(*bounds)
    data.data[data.data == -9999] = 0

    if plot:
        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': 12})
        data.plot(ax=ax, cbar_kwargs={'label': "Normalized Green Value"})
        ax.set_title('')
        ax.set_xlabel('RD New X coordinate')
        ax.set_ylabel('RD New Y coordinate')
        fig.savefig("Images/Greenery Heatmap.png", bbox_inches='tight', dpi=300)
        ax.set_title("Greenness of Delft from Dutch 'Groenkaart'")
        fig.show()
    
    return data


def assign_groen_data_to_network(network: gpd.GeoDataFrame, data) -> gpd.GeoDataFrame:

    gdf = vectorize(data)

    network['green'] = [0.0] * len(network)
    for index, section in network.iterrows():

        bounds = section['geometry'].buffer(5).bounds
        possible = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()

        matches = possible[possible.intersects(section['geometry'].buffer(5))]
        if len(matches) > 0:

            network.loc[index, 'green'] = np.mean(list(matches['myData'].values))
    
    return network

def plot_groen_data_comparison(which_set: str, network: gpd.GeoDataFrame, cluster_colors: dict[str, str]) -> None:
    if which_set not in ['hierarchical', 'all']:
        raise ValueError("which_set can only be 'hierarchical' or 'all'")
    
    if which_set == 'hierarchical':
        fig, axs = plt.subplots(1, 1, sharey=True, figsize=(5, 6))
        plt.rcParams.update({'font.size': 14})
        clustering_cols = ['cluster_hr']

        network_hr = network[pd.notna(network['cluster_hr'])]
        grouped = network_hr.loc[:,['cluster_hr', 'green']] \
                                .groupby(['cluster_hr']) \
                                .median() \
                                .sort_values(by='green', ascending=False)

        for ax, clustering_col in zip([axs], clustering_cols):
            network_i = network[pd.notna(network[clustering_col])]

            #sns.boxplot(x=network_i[clustering_col], y=network_i['green'], ax=ax,
            #            order=grouped.index, width=0.67, linewidth=1.1, notch=True,
            #            palette=cluster_colors, medianprops=dict(color="red"))
            
            sns.violinplot(x=network_i[clustering_col], y=network_i['green'], ax=ax,
               order=grouped.index, width=0.67, linewidth=1.1,
               palette=cluster_colors)

            ax.set_xlabel('')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Normalized green value')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
            ax.grid(axis='y')

        plt.tight_layout()
        fig.savefig("Images/Greenery Boxplot Hierarchical.png", bbox_inches='tight', dpi=300)
        plt.show()


    else:  # which_set == 'all':
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 6))
        plt.rcParams.update({'font.size': 14})
        clustering_cols = ['cluster_km', 'cluster_hr', 'cluster_gm']
        titles = ['K-Means', 'Hierarchical', 'Gaussian Mixture']

        network_km = network[pd.notna(network['cluster_km'])]
        grouped = network_km.loc[:,['cluster_km', 'green']] \
                                .groupby(['cluster_km']) \
                                .median() \
                                .sort_values(by='green', ascending=False)

        for ax, clustering_col, title in zip(axs, clustering_cols, titles):
            network_i = network[pd.notna(network[clustering_col])]

            sns.boxplot(x=network_i[clustering_col], y=network_i['green'], ax=ax, order=grouped.index, width=0.67, linewidth=1.1, notch=True,
                        palette=cluster_colors, medianprops=dict(color="red"))

            ax.set_title(title)
            ax.set_xlabel('')
            if ax == axs[1]:
                ax.set_xlabel('Cluster')
            if ax == axs[0]:
                ax.set_ylabel('Normalized green value')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
            ax.grid(axis='y')

        plt.tight_layout()
        fig.savefig("Images/Greenery Boxplot.png", bbox_inches='tight', dpi=300)
        plt.show()

