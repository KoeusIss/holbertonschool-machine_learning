#!/usr/bin/env python3
"""Clustering module"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means

    Arguments:
        X {np.ndarray} -- Containing the data points
        k {int} -- Is the number of clusters

    Returns:
        tuple -- containing the centroids and the index for each data point
        belongs.
    """
    kmeans = sklearn.cluster.KMeans(k)
    model = kmeans.fit(X)
    return model.cluster_centers_, model.labels_
