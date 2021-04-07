#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model

    Arguments:
        Observation {np.ndarray} -- Contains the index of observation of shape
        (T,) where T is the number of observations
        Emission {np.ndarray} -- Contains the emission probability of
        a specefic observation given a hidden state, with shape (N, M) where
        N is the number of hidden states and M is all the possible observation.
        Transition {np.ndarray} -- Contains the transition probabilites with
        shape (N, N) where N is the number of states
        Initial {np.ndarra} -- Contains the starting probability in particular
        state

    Returns:
        tuple(float, np.ndarray) -- Contains the liklihood of the obsrvations
        given the model, and the path probabilities
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    T, = Observation.shape
    N, M = Emission.shape

    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if np.any(np.sum(Transition, axis=1) != 1):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    if np.sum(Initial) != 1:
        return None, None

    # Initialization
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Recursion
    for t in range(1, T):
        for s in range(N):
            obs = Observation[t]
            F[s, t] = np.sum(Emission[s, obs] * F[:, t-1] * Transition[:, s])

    # Termination
    P = np.sum(F[:, T - 1])
    return P, F
