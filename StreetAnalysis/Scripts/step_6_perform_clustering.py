import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from scipy.cluster.hierarchy import dendrogram
from matplotlib.colors import to_rgb, ListedColormap
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from Scripts.clustering_functions import scale_data, dim_reduction, pca_kmeans, pca_hierarchical, \
    pca_gaussianmixture, generate_color_codes


def perform_clustering(features_array: np.ndarray, features_ids: list[str], n_cluster: int,
                       n_components: int, random_state: int) -> pd.DataFrame:
    features_scaled = scale_data(features_array)
    print("Scaled features to between 0 and 1")

    features_dim_reduced = dim_reduction(features_scaled, n_components=n_components, random_state=random_state)
    print("Reduced feature dimensions")

    clustering_km, _ = pca_kmeans(features_dim_reduced, n_cluster, random_state=random_state)
    print("Performed Kmeans clustering")
    clustering_hr, linked = pca_hierarchical(features_dim_reduced, n_cluster)
    print("Performed Hierarchical clustering")
    clustering_gm, _ = pca_gaussianmixture(features_dim_reduced, n_cluster, random_state=random_state)
    print("Performed Gaussian Mixture clustering")

    results_transposed = np.transpose([features_ids, clustering_km.labels_, clustering_hr.labels_, clustering_gm])
    results_df = pd.DataFrame(results_transposed, columns=['section_id', 'cluster_km', 'cluster_hr', 'cluster_gm'])
    results_df['cluster_km'] = results_df['cluster_km'].astype(int)
    results_df['cluster_hr'] = results_df['cluster_hr'].astype(int)
    results_df['cluster_gm'] = results_df['cluster_gm'].astype(int)
    print("Clusterings completed, combined results")

    return results_df, features_dim_reduced, linked


def visualize_tsne(features_array: np.ndarray, clustering_results: pd.DataFrame, random_state: int=0,
                   color_offset: float=0.4, save_stage: str|None=None) -> None:
    """Use TSNE to visualize the high dimension features with labeling to show their clusters""" 
    
    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    colors = generate_color_codes(6, intital_offset=color_offset)
    #colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    colors = [to_rgb(color) for color in colors]
    colormap = ListedColormap(colors)

    labels_list = [clustering_results['cluster_km'], clustering_results['cluster_hr'],
                   clustering_results['cluster_gm']]
    titles = ['Kmeans', 'Hierarchical', 'Gaussian Mixture']

    for ax, labels, title in zip(axs, labels_list, titles):
        tsne = TSNE(n_components=2, random_state=random_state)
        feat_tsne = tsne.fit_transform(features_array)

        ax.scatter(feat_tsne[:,0], feat_tsne[:,1], c=labels, cmap=colormap, alpha=0.7)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_stage is not None:
        if save_stage == 'before':
            plt.savefig(r'Images\TSNE before.png', bbox_inches='tight', dpi=300)
        if save_stage == 'after':
            plt.savefig(r'Images\TSNE after.png', bbox_inches='tight', dpi=300)

    plt.suptitle("Dimensionally Condensed Clustering Results")
    plt.show()


def plot_hierarchical_dendrogram(model: AgglomerativeClustering, n_clusters: int) -> None:
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, truncate_mode='lastp', p=n_clusters)
    plt.title('Hierarchical Clustering Results')
    plt.ylabel("Distance")
    plt.xlabel("Amount of Sections per cluster")


def plot_heatmaps(clustering_results: pd.DataFrame, cmap_name: str='winter', save_stage: str|None=None) -> None:
    clustering_results = clustering_results.copy()

    km_vs_hr = clustering_results.drop(columns=['cluster_gm']).groupby(['cluster_km', 'cluster_hr'], group_keys=False).count()
    km_vs_hr = km_vs_hr.reset_index().pivot(columns='cluster_km',index='cluster_hr',values='section_id')
    km_vs_gm = clustering_results.drop(columns=['cluster_hr']).groupby(['cluster_km', 'cluster_gm'], group_keys=False).count()
    km_vs_gm = km_vs_gm.reset_index().pivot(columns='cluster_km',index='cluster_gm',values='section_id')
    hr_vs_gm = clustering_results.drop(columns=['cluster_km']).groupby(['cluster_hr', 'cluster_gm'], group_keys=False).count()
    hr_vs_gm = hr_vs_gm.reset_index().pivot(columns='cluster_hr',index='cluster_gm',values='section_id')

    fig, axs = plt.subplots(1, 3, figsize=(6,5))
    fig.set_figheight(5)
    fig.set_figwidth(18)

    base_color = plt.get_cmap(cmap_name)(0)
    titles = ['Kmeans vs Hierachical', 'Kmeans vs Guassian Mixture', 'Hierachical vs Gaussian Mixture']

    min_overlap = 0
    max_overlap = np.max([np.nanmax(km_vs_hr.values), np.nanmax(km_vs_gm.values), np.nanmax(hr_vs_gm.values)])

    for i, (ax, df, title) in enumerate(zip(axs, [km_vs_hr, km_vs_gm, hr_vs_gm], titles)):
        heatmap_i = sns.heatmap(df, ax=ax, cbar=False, cmap=cmap_name, vmin=min_overlap, vmax=max_overlap)
        heatmap_i.set_facecolor(base_color)
        ax.invert_yaxis()
        ax.set_title(title) 

    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.92,0.11,0.01,0.77])
    cax.set_frame_on(False)
    fig.colorbar(im, cax=cax)

    if save_stage is not None:
        if save_stage is 'before':
            plt.savefig(r'Images\Heatmap before.png', bbox_inches='tight', dpi=300)
        if save_stage is 'after':
            plt.savefig(r'Images\Heatmap after.png', bbox_inches='tight', dpi=300)


def renumber_clusters(clustering_results: pd.DataFrame, dominant: str, non_dominant: str, n_cluster: int) -> pd.DataFrame:

    sub_per_dom = clustering_results.drop(clustering_results.columns.difference([dominant, non_dominant, 'section_id']),
                                          axis=1).copy()
    sub_per_dom = sub_per_dom.groupby([dominant, non_dominant], group_keys=False).count()
    sub_per_dom = sub_per_dom.sort_values(by='section_id', ascending=False).reset_index()

    # Find best corresponding cluster numbers from dominant clustering type for submissive clustering type cluster numbers
    recluster = dict.fromkeys(range(n_cluster))
    taken = []
    for _, count in sub_per_dom.iterrows():
        if recluster[count[non_dominant]] is None and count[dominant] not in taken:
            recluster[count[non_dominant]] = count[dominant]
            taken.append(count[dominant])

    # Fill None clusters with random not taken clusters
    not_taken = [cluster for cluster in range(n_cluster) if cluster not in taken]
    for cluster_sub, cluster_dom in recluster.items():
        if cluster_dom is None:
            recluster[cluster_sub] = not_taken[0]
            not_taken.pop()

    # Change numbers to correct corresponding
    old_results = clustering_results.copy()
    new_results = clustering_results.copy()
    for cluster_sub, cluster_dom in recluster.items():
        new_results.loc[old_results[non_dominant] == cluster_sub, non_dominant] = cluster_dom

    return new_results, recluster