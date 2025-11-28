import numpy as np
import colorsys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram


def create_feature_vector_array(network_feature_vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    network_ids = []
    network_features = []
    for id, features in network_feature_vectors.items():
        network_ids.append(id)
        network_features.append(features)
    network_features = np.array(network_features)

    return network_ids, network_features


def scale_data(features_array: np.ndarray) -> np.ndarray:
    """Scale the data to between 0 and 1"""
    scaler = MinMaxScaler()
    
    return scaler.fit_transform(features_array)


# TODO: Rename variance to n_components and provide explanation
def dim_reduction(features_array: np.ndarray, n_components: int|float, random_state: int) -> np.ndarray:
    """Perform dimensionality reduction on a network's features array"""
    pca = PCA(n_components=n_components, svd_solver='full', random_state=random_state)
    pca.fit(features_array)

    dim_reduced_features = pca.transform(features_array)
    return dim_reduced_features


def pca_kmeans(features_array: np.ndarray, n_clusters: int, random_state: int) -> tuple[KMeans, float]:
    """Perform kmeans clustering on a scaled and dimensionality reduced network's features array"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(features_array)

    inertia = kmeans.inertia_
    return kmeans, inertia


def calc_silhouette(features_array: np.ndarray, kmeans_object: KMeans, sample_size: int|None) -> float:
    """Calculate the silhouette scroe for a Kmeans clustering result"""

    if len(set(kmeans_object.labels_)) > 1:
        # Calculatted using euclidean distance
        try:
            score = silhouette_score(features_array, kmeans_object.labels_, sample_size=sample_size)
        except ValueError:
            score = calc_silhouette(features_array, kmeans_object, sample_size)
    
    # TODO: Do we want this else statement here? The plot will break if we dont get a score, so we should just throw an error
    else:
        score = np.nan

    return score


def pca_hierarchical(features_array: np.ndarray, n_clusters: int) -> AgglomerativeClustering:
    """Perform the hierarchical clustering for features after dimensionality reduction."""
    # Using 'ward' linkage makes AgglomerativeClustering automatically use 'euclidean' as metric
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_distances=True)
    labels = hierarchical.fit(features_array)
    linked = linkage(features_array, method='ward')
    
    return labels, linked


def pca_gaussianmixture(features_array: np.ndarray, n_clusters: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform the Gaussian Mixture clustering for features after dimensionality reduction."""
    # Covariance_type = 'full' ensure that each component has its own covariance matrix, which we want as the components are distinct from each other.
    gm = GaussianMixture(n_components = n_clusters, covariance_type="full", random_state=random_state).fit(features_array)
    gm_clusters = gm.predict(features_array)
    gm_prob = gm.predict_proba(features_array)

    return gm_clusters, gm_prob


def generate_color_codes(number_of_colors: int, intital_offset: float=0.4) -> list[tuple[float, float, float]]:
    golden_ratio_conjugate = 0.618033988749895
    h = intital_offset

    colors = []
    for i in range(number_of_colors):
        h += golden_ratio_conjugate
        h = h % 1
        rbg_color = colorsys.hsv_to_rgb(h, 0.9, 0.85)
        colors.append('#%02x%02x%02x' % (int(rbg_color[0]*255), int(rbg_color[1]*255), int(rbg_color[2]*255)))

    return colors