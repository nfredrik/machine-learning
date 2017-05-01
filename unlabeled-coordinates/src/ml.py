"""Machine learning related utilities."""

from typing import List, Tuple
import random

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


Vector = List[float]
Boundaries = Tuple[float, float, float, float]
Cluster = List[Vector]


class KMeansClassifier:
    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters
        self._clusters = []

    def fit(self, data: List[Vector]) -> None:
        """Calculate clusters based on given data."""
        centroids = self._random_centroids(data)
        while self._iterate(data, centroids):
            centroids = centroids_in(self._clusters)

    def _iterate(self, data: List[Vector], centroids: List[Vector]) -> bool:
        """
        Returns:
            True if one should keep iterating cluster recalculation,
            False otherwise.
        """
        clusters = classify_points(data, centroids)
        if cluster_lists_equal(clusters, self._clusters):
            return False
        self._clusters = clusters
        return True

    def _random_centroids(self, data: List[Vector]) -> List[Vector]:
        bounds = data_boundaries(data)
        return [random_centroid_in(bounds) for _ in range(self.n_clusters)]


def cluster_lists_equal(clusters1: List[Cluster],
                        clusters2: List[Cluster]) -> bool:
    # TODO: clusters can be equal even if the vector order is not the same.
    return np.array_equal(clusters1, clusters2)


def classify_points(data: List[Vector],
                    centroids: List[Vector]) -> List[Cluster]:
    clusters = [[] for _ in range(len(centroids))]
    for v in data:
        d = euclidean_distances(centroids, [v]).flatten()
        clusters[d.argmin()].append(v)
    return clusters


def centroids_in(clusters: List[Cluster]) -> List[Vector]:
    """Finds centroids in multiple clusters."""
    return [np.mean(cluster, axis=0) for cluster in clusters]


def random_centroid_in(bounds: Boundaries) -> Tuple[float, float]:
    return (
        random.uniform(bounds[0], bounds[2]),
        random.uniform(bounds[1], bounds[3]),
    )


def data_boundaries(data: List[Vector]) -> Tuple[float, float, float, float]:
    return [
        min(map(lambda item: item[0], data)),
        min(map(lambda item: item[1], data)),
        max(map(lambda item: item[0], data)),
        max(map(lambda item: item[1], data)),
    ]
