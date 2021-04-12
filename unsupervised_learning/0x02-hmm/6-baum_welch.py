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
    T, = Observation.shape
    N, M = Emission.shape
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
    T, = Observation.shape
    N, M = Emission.shape
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


def expectation(Observations, Transition, Emission, forward, backward):
    """Performs the Expectation step

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
        forward {tuple} -- Contains the liklihood of the obsrvations
        given the model, and the path probabilities
        backward {tuple} -- Contains the liklihood of the obsrvations
        given the model, and the path probabilities

    Returns:
        tuple(np.ndarray, np.ndarray) -- Contain the gamma and xi
    """
    T, = Observations.shape
    N, M = Emission.shape

    P_o_given_model, F = forward
    _, B = backward

    xi = np.zeros((N, N, T - 1))
    for t in range(T - 1):
        obs = Observations[t + 1]
        for i in range(N):
            for j in range(N):
                xi[i, j, t] = (
                    F[i, t] *
                    Transition[i, j] *
                    Emission[j, obs] *
                    B[j, t + 1]
                ) / P_o_given_model
    gamma = (F * B) / P_o_given_model
    return gamma, xi


def maximization(Observations, gamma, xi, dimension):
    """Performs the Maximization step

    Arguments:
        Observations {np.ndarray} -- Contains the index of the obsevation
        gamma {np.ndarray} -- Contains the gamma array
        xi {np.ndarray} -- Contains the xi array
        dimension {tuple} -- Contains dimension of the emission

    Returns:
        tuple(np.ndarray) -- Contain the emission and Transition
    """
    N, M = dimension
    T, = Observations.shape
    Transition = np.sum(xi, axis=2) / np.sum(
        gamma[:, :T-1], axis=1
        )[..., np.newaxis]
    numerator = np.zeros((N, M))
    for k in range(M):
        numerator[:, k] = np.sum(gamma[:, Observations == k], axis=1)
    Emission = numerator / np.sum(gamma, axis=1)[..., np.newaxis]
    return Transition, Emission


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the BAUM WELCH algorithm

    Arguments:
        Observations {np.ndarray} -- Contains the index of observation
        Transition {np.ndarray} -- Contains the transition matrix
        Emission {np.ndarray} -- Contains the emission probabilities
        Initial {np.ndarray} -- Contains the initial state

    Keyword Arguments:
        iterations {int} -- Number of iterations (default: {1000})

    Returns:
        tupl(np.ndarray, np.ndarray) -- The estimation for Transition and
        Emission
    """
    if not isinstance(Observations, np.ndarray) or\
            len(Observations.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    T, = Observations.shape
    N, M = Emission.shape

    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if np.any(np.sum(Transition, axis=1) != 1):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    if np.sum(Initial) != 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    for i in range(iterations):
        gamma, xi = expectation(
            Observations,
            Transition,
            Emission,
            forward(Observations, Emission, Transition, Initial),
            backward(Observations, Emission, Transition, Initial)
        )
        Transition, Emission = maximization(Observations, gamma, xi, (N, M))

    return Transition, Emission
