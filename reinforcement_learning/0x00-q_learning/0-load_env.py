#!/usr/bin/env python3
"""Q-learning module"""
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads a pre-made FrozenLakeEnv environment

    Keyword Arguments:
        desc {List[List]} -- Contains custom description (default: {None})
        map_name {string} -- Contains the pre-made map (default: {None})
        is_slippery {bool} -- Determines if the ice is slippery
        (default: {False})

    Returns:
        FrozenLakeEnv -- FrozenLakeEnv instance
    """
    env = FrozenLakeEnv(desc, map_name, is_slippery)
    return env
