#!/usr/bin/env python3
"""Clustering module"""
import sklearn.cluster


def kmeans(X, k):
    kmeans = sklearn.cluster.KMeans(k)
    model = kmeans.fit(X)
    return model.cluster_centers_, model.labels_
