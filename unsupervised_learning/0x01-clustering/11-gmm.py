#!/usr/bin/env python3
import sklearn.mixture


def gmm(X, k):
    _gmm = sklearn.mixture.GaussianMixture(k)
    model = _gmm.fit(X)
    classes = _gmm.predict(X)
    bic = _gmm.bic(X)
    return model.weights_, model.means_, model.covariances_, classes, bic
