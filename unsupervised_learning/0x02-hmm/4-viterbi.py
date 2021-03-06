#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Performs The Viterbi Algorithm

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
        tuple(list, float) -- The most likely sequence of hidden states,
        The probability of obtaining the path sequence
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
    v = np.zeros((N, T))
    backpointers = np.zeros((N, T))
    v[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Recursion
    for t in range(1, T):
        for s in range(N):
            obs = Observation[t]
            v[s, t] = np.max(v[:, t-1] * Emission[s, obs] * Transition[:, s])
            backpointers[s, t] = np.argmax(
                v[:, t-1] * Emission[s, obs] * Transition[:, s], 0
            )

    # Termination
    P = np.max(v[:, T - 1])
    best_path_pointer = np.argmax(v[:, T - 1])
    path = [best_path_pointer]
    for t in range(T - 1, 0, -1):
        best_path_pointer = int(backpointers[best_path_pointer, t])
        path.append(best_path_pointer)
    return path[::-1], P
