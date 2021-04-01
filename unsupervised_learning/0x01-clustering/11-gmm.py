#!/usr/bin/env python3
import sklearn.mixture


def gmm(X, k):
    """Performs the GMM

    Arguments:
        X {np.ndarray} -- Containing the data points
        k {int} -- Is the number of clusters

    Returns:
        tuple -- priors, means, covariances, clss, bic
    """
    _gmm = sklearn.mixture.GaussianMixture(k)
    model = _gmm.fit(X)
    clss = _gmm.predict(X)
    bic = _gmm.bic(X)
    return model.weights_, model.means_, model.covariances_, clss, bic
