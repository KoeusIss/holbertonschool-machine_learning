#!/usr/bin/env python3
"""Clustering module"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering

    Arguments:
        X {np.ndarray} -- Containing the data points
        dist {int} -- Is the maximum cophenetic distances

    Returns:
        np.ndarray -- cluster indices
    """
    links = scipy.cluster.hierarchy.linkage(X, method='ward')
    _plot = scipy.cluster.hierarchy.dendrogram(links, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(
        links,
        t=dist,
        criterion='distance'
    )
    plt.show()
    return clss
