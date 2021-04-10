#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden markov model

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

    # initialization
    B = np.zeros((N, T))
    B[:, T - 1] = 1

    # Recursion
    for t in range(T - 2, -1, -1):
        for s in range(N):
            obs = Observation[t + 1]
            B[s, t] = np.sum(B[:, t+1] * Emission[:, obs] * Transition[s, :])
    P = np.sum(B[:, 0] * Emission[:, Observation[0]] * Initial.T)
    return P, B
