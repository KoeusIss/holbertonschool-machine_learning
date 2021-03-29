#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def initialize(X, k):
    try:
        n, d = X.shape
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return np.random.uniform(
            low=X_min,
            high=X_max,
            size=(k, d)
        )
    except Exception:
        return None
