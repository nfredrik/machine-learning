from typing import List, Dict

import numpy as np


Vector = List[float]
Cluster = List[Vector]


def classification_errors(initial_data: List[Cluster],
                          clustered_data: List[Cluster]) -> int:
    """
    Returns:
        Number of mismatching vectors in the clusters.
    """
    errors = 0
    index_map = cluster_map(initial_data, clustered_data)
    for i, cl1 in enumerate(initial_data):
        cl2 = clustered_data[index_map[i]]
        errors += len(cl1) - cluster_matches(cl1, cl2)
    return errors


def cluster_map(clusters1: List[Cluster],
                clusters2: List[Cluster]) -> Dict[int, int]:
    """Maps two cluster lists by similarities.

    Clustering algorithms might return clusters in a different order than the
    given labeled clusters. Thus before comparing clusters we need to map
    which cluster is most likely which in another list.
    """
    cl_map = {}
    for i, cl in enumerate(clusters1):
        matches = [cluster_matches(cl, cl2) for cl2 in clusters2]
        cl_map[i] = np.argmax(matches)
    return cl_map


def cluster_matches(cluster1: Cluster, cluster2: Cluster) -> int:
    """
    Returns:
        number of points that match in both clusters.
    """
    return sum(map(lambda _: 1, filter(lambda v: v in cluster2, cluster1)))
