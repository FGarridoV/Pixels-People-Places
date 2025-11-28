import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from Scripts.clustering_functions import scale_data, dim_reduction, pca_kmeans, calc_silhouette


def iterate_pca_means(features_array: np.ndarray, iterations: int, variances: list[float], max_clusters: int,
                      random_state: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Method to iterate over different cluster numbers, components for the pca and sample sized for the 
    silhouette score. For each iteration, differnent random states are used to slightly randomize the process.

    Parameters:
    features: the original (unscaled) features
    iterations: Number of to perform the entire pca_means
    variances: List of amounts of variance to perform clustering with.
    max_clusters: Perform clustering from one up to and including this number of clusters.
    random_state: Locks random state of dimension reduction used for determining the number of components.

    Returns:
    kmeans_array: 3d numpy array with kmeans objects of all performed clusterings 
                    size: (iterations x len(cluster_list) x len(variance))
    inertia_array: 3d numpy array with inertias of each clustering 
                    size: (iterations x len(cluster_list) x len(variance))
    silhouette_array: 3d numpy array with all silhouette scores 
                    size: (iterations x len(cluster_list) x len(variance))
    components_list: 1d numpy array containing the number of components for each variance
                    size: (len(variance))
    """
    print(f"Starting pca kmeans iteration with {iterations} iterations, with {len(variances)} variances per \
iteration and {max_clusters} number of clusters per variance")
    features_scaled = scale_data(features_array) # scale features

    print(f"Getting component list for given variances")
    components_list = []
    for variance in variances:
        dim_reduced_features = dim_reduction(features_scaled, variance, random_state=random_state)
        components_list.append(dim_reduced_features.shape[1])
        print(f"\t{dim_reduced_features.shape[1]} components for variance {variance}")

    # Create arrays to store the results
    kmeans_array = np.empty((iterations, len(components_list), max_clusters-1), dtype=object)
    inertia_array = np.zeros((iterations, len(components_list), max_clusters-1))
    silhouette_array = np.zeros((iterations, len(components_list), max_clusters-1))

    # Perform iteration
    for iteration in range(iterations):
        print(f"\nIteration {iteration+1}")

        for var_index, variance in enumerate(variances):
            print(f"\tVariance: {variance}")
            
            # Dimensionality reduction
            dim_reduced_features = dim_reduction(features_scaled, components_list[var_index], random_state=iteration)
            
            for n_clusters in range(2, max_clusters+1):
                # print(f"\t{n_clusters} clusters")
                
                # Clustering
                kmeans_object, inertia = pca_kmeans(dim_reduced_features, n_clusters, random_state=iteration)
                kmeans_array[iteration, var_index, n_clusters-2] = kmeans_object
                inertia_array[iteration, var_index, n_clusters-2] = inertia
                
                # Determing silhouette scores
                score = calc_silhouette(dim_reduced_features, kmeans_object, None)
                silhouette_array[iteration, var_index, n_clusters-2] = score

    return kmeans_array, inertia_array, silhouette_array, components_list


def plot_silhouette(inertia_array: np.ndarray, silhouette_array: np.ndarray, variances: list[float],
                    components_list: list[float], max_clusters: int, network_name: str, save:bool=False) -> None:
    """Plot the silhouette score for the kmeans clustering iterations"""

    plt.figure(figsize=(10, 6))

    # TODO: Update these colors to something more fitting of the overall colorpallete?
    color_list = np.array([['salmon', 'lightblue', 'lightgreen', 'orange'], 
                   ['firebrick', 'midnightblue', 'darkgreen', 'darkgoldenrod'],
                   ['sienna', 'aqua', 'palegreen', 'gold']])

    for iteration in range(len(inertia_array)):
        for var_index, variance in enumerate(variances):
            if iteration == 0:
                plt.plot([], [], 
                            label=f'{variance * 100:.0f}% variance, {components_list[var_index]} components', 
                            color=color_list[1, var_index])
                
            plt.plot(range(2, max_clusters+1), silhouette_array[iteration, var_index, :],
                     color=color_list[1, var_index], alpha=0.6)

    plt.xlabel('Number of clusters')
    plt.xticks(range(2, max_clusters+1))
    plt.ylabel('Silhouette score')
    
    # TODO: Somehow make this ymax automatic?
    if np.nanmax(silhouette_array) < 0.15:
        ymax = np.nanmax(silhouette_array)
    else:
        ymax = 0.17

    plt.ylim(np.nanmin(silhouette_array), ymax)
    plt.legend()

    if save:
        plt.savefig(r'Images\Silhouette Score.png', bbox_inches='tight', dpi=300)

    plt.title(f'Silhouette Score of PCA Kmeans Clustering using {network_name} Features')



def plot_inertia(inertia_array: np.ndarray, variances: list[float], components_list: list[int],
                 max_clusters: int, network_name: str, save:bool=False) -> None:
    """Plot the inertia for the kmeans clustering iterations"""

    plt.figure(figsize=(10, 6))

    # TODO: Update these colors to something more fitting of the overall colorpallete?
    color_list = np.array([['salmon', 'lightblue', 'lightgreen', 'orange'], 
                   ['firebrick', 'midnightblue', 'darkgreen', 'darkgoldenrod'],
                   ['sienna', 'aqua', 'palegreen', 'gold']])

    for iteration in range(len(inertia_array)):
        for var_index, variance in enumerate(variances):
            if iteration == 0:
                plt.plot([], [], label=f'{variance * 100:.0f}% variance, {components_list[var_index]} components', 
                         color=color_list[1, var_index])
            
            plt.plot(range(2, max_clusters+1), inertia_array[iteration, var_index, :],
                     color=color_list[1, var_index], alpha=0.6)

    plt.xlabel('Number of clusters')
    plt.xticks(range(2, max_clusters+1))
    plt.ylabel('Inertia')
    plt.legend()

    if save:
        plt.savefig(r'Images\Inertia.png', bbox_inches='tight', dpi=300)

    plt.title(f'PCA KMeans Clustering Inertia of {network_name} Features')



def determine_elbow_point(inertia_array: np.ndarray, variance: int, max_clusters: int, factor: float=1,
                          plot: bool=False, variances: np.ndarray=None, components_list: list[int]= None,
                          save: bool=False) -> None:

    elbow_point_total = 0
    elbow_begin_total = 0
    elbow_end_total = 0

    for iteration in range(len(inertia_array)):
        deltas = [np.abs(inertia_array[iteration, variance, i] - \
                         inertia_array[iteration, variance, i-1]) \
                  for i in range(1, max_clusters-2)]

        for i in range(1, len(deltas)-2):
            if np.abs(deltas[-1] - deltas[i]) < np.abs([deltas[i]] - deltas[0]):
                elbow_delta_index = i
                elbow_point_total += (elbow_delta_index+3) / len(inertia_array)
                break

        for j in range(1, elbow_delta_index):
            if np.abs(deltas[elbow_delta_index] - deltas[j]) < np.abs([deltas[j]] - deltas[0]):
                elbow_begin_total += (j+3) / len(inertia_array)
                break

        for k in range(elbow_delta_index+1, len(deltas)-2):
            if np.abs(deltas[-1] - deltas[k]) < np.abs([deltas[k]] - deltas[elbow_delta_index]):
                elbow_end_total += (k+3) / len(inertia_array)
                break
                    
    # TODO: Do this plotting differently, function has to many arguments now
    if plot is False and (variances is None or components_list is None):
        raise ValueError("Plot set to True but variances array and/or components_list not given")
    
    elif plot: 
        plot_inertia(inertia_array, variances=[variances[variance]], components_list=components_list, max_clusters=max_clusters, network_name='test')
        plt.axvspan(elbow_begin_total, elbow_end_total, color='lightblue', alpha=0.5, label='Elbow region')
        plt.axvline([elbow_point_total], label=f"Elbow point ({elbow_point_total:.3f})")
        plt.legend()

        if save:
            plt.title(label='')
            plt.savefig(r'Images\Elbow Point.png', bbox_inches='tight', dpi=300)
    
    print(f"Considering the elbow point, the optimal number of clusters is {elbow_point_total:.2f}")