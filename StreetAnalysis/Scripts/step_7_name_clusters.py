import random
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import altair as alt
from keras.preprocessing.image import load_img
from Scripts.clustering_functions import generate_color_codes

# %matplotlib inline
alt.data_transformers.disable_max_rows()



def assign_clusters_to_geoms(network: gpd.GeoDataFrame, clustering_results: pd.DataFrame) -> gpd.GeoDataFrame:
    network = network.copy()
    network['ID'] = network['ID'].astype('str')
    network['cluster_km'] = [None] * len(network)
    network['cluster_hr'] = [None] * len(network)
    network['cluster_gm'] = [None] * len(network)

    for index, section in network.iterrows():
        if section['ID'] in clustering_results['section_id'].values:
            section_results = clustering_results[clustering_results['section_id'] == section['ID']]
            network.loc[index, 'cluster_km'] = section_results['cluster_km'].values[0]
            network.loc[index, 'cluster_hr'] = section_results['cluster_hr'].values[0]
            network.loc[index, 'cluster_gm'] = section_results['cluster_gm'].values[0]
    
    return network

def plot_hierarchical_clustering(prepped_network: gpd.GeoDataFrame, cluster_names: list[int|str],
                                 random_color_state: float=0.4, figure_type: str='normal',
                                 save_name: str=None) -> tuple[alt.Chart, dict[int | str, tuple[float, float, float]]]:
    if figure_type not in ['normal', 'cover']:
        raise ValueError("figure_type can only be 'normal' or 'cover'")

    domain_edges = cluster_names

    range_colors = generate_color_codes(len(domain_edges), random_color_state)
    color_export = {cluster: color for cluster, color in zip(domain_edges, range_colors)}

    background = alt.Chart(prepped_network).mark_geoshape(
        filled=False,
        stroke='darkgray',
        strokeWidth=2.5
    ).encode(
        tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N']
    ).properties(
        width=400,
        height=550,
    ).project(
        'mercator'
    )

    if figure_type == 'normal':
        hierarch_edges = alt.Chart(prepped_network).mark_geoshape(
            filled=False,
            strokeWidth=2.5
        ).encode(
            tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N'],
            color = alt.Color('cluster_hr:N', title="Clusters").scale(domain=domain_edges, range=range_colors)
        ).properties(
            width=400,
            height=550,
        ).project(
            'mercator'
        )
    
    else:  # figure_type  == 'cover':
        hierarch_edges = alt.Chart(prepped_network).mark_geoshape(
            filled=False,
            strokeWidth=2.5
        ).encode(
            tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N'],
            color = alt.Color('cluster_hr:N', title="Clusters", legend=None).scale(domain=domain_edges, range=range_colors)
        ).properties(
            width=400,
            height=550,
        ).project(
            'mercator'
        )

    chart = background + hierarch_edges
    chart = chart.configure_legend(labelFontSize=10, titleFontSize=10)
    if save_name:
        chart.save(f"Images/{save_name}.png", ppi=300)
    
    return chart


def plot_all_methods_clustering(prepped_network: gpd.GeoDataFrame, cluster_names: list[int|str],
                                random_color_state: float=0.4,
                                save_name: str=None) -> tuple[alt.Chart, dict[int | str, tuple[float, float, float]]]:
    domain_edges = cluster_names

    range_colors = generate_color_codes(len(domain_edges), random_color_state)
    color_export = {cluster: color for cluster, color in zip(domain_edges, range_colors)}

    background = alt.Chart(prepped_network).mark_geoshape(
        filled=False,
        stroke='darkgray',
        strokeWidth=2.5
    ).encode(
        tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N']
    ).properties(
        width=400,
        height=550,
    ).project(
        'mercator'
    )

    k_means_edges = alt.Chart(prepped_network).mark_geoshape(
        filled=False,
        strokeWidth=2.5
    ).encode(
        tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N'],
        color = alt.Color('cluster_km:N', title="Clusters").scale(domain=domain_edges, range=range_colors)
    ).properties(
        width=400,
        height=550,
        title='K-means'
    ).project(
        'mercator'
    )

    hierarch_edges = alt.Chart(prepped_network).mark_geoshape(
        filled=False,
        strokeWidth=2.5
    ).encode(
        tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N'],
        color = alt.Color('cluster_hr:N', title="Clusters").scale(domain=domain_edges, range=range_colors)
    ).properties(
        width=400,
        height=550,
        title='Hierarchical'
    ).project(
        'mercator'
    )

    gaussian_edges = alt.Chart(prepped_network).mark_geoshape(
        filled=False,
        strokeWidth=2.5
    ).encode(
        tooltip=['ID:N', 'cluster_km:N', 'cluster_hr:N', 'cluster_gm:N'],
        color = alt.Color('cluster_gm:N', title="Clusters").scale(domain=domain_edges, range=range_colors)
    ).properties(
        width=400,
        height=550,
        title='Gaussian Mixture'
    ).project(
        'mercator'
    )

    chart = background + k_means_edges | background + hierarch_edges | background + gaussian_edges
    chart = chart.configure_legend(labelFontSize=16, titleFontSize=16)
    chart = chart.configure_title(fontSize=16)
    if save_name:
        chart.save(f"Images/{save_name}.png", ppi=300)

    return chart, color_export


def assign_images_to_clusters(clustering_results: pd.DataFrame, n_cluster: int,
                              image_assignments: dict[str, list[str]]) -> dict[int, list[str]]:
    cluster_images = {key: [] for key in range(n_cluster)}

    for _, section_clusters in clustering_results.iterrows():
        cluster_images[section_clusters['cluster_km']].extend(image_assignments[section_clusters['section_id']])
        cluster_images[section_clusters['cluster_hr']].extend(image_assignments[section_clusters['section_id']])
        cluster_images[section_clusters['cluster_gm']].extend(image_assignments[section_clusters['section_id']])
    
    return cluster_images


def sample_cluster_images(cluster_images: dict[int, list[str]], images_folder: str,
                          cluster: int, cluster_name: str=None, nrows: int=3, ncols: int=7) -> None:

    image_sample = random.sample(cluster_images[cluster], int(nrows*ncols))

    images = [load_img(f"{images_folder}/{image}", target_size=(224,224)) for image in image_sample]

    f, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows))
    axes = axes.flatten()[:len(images)]
    
    for image, ax in zip(images, axes): 
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    f.savefig(f"Images/Images Cluster {cluster}.png", bbox_inches='tight')

    if cluster_name is not None:
        f.suptitle(f"Image Sample for cluster '{cluster_name}'")
    else:
        f.suptitle(f"Image Sample for cluster {cluster}")

    plt.show()
