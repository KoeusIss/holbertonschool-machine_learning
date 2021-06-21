#!/usr/bin/env python3
"""Q-learning module"""
import gym


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
    FrozenLakeEnv = gym.envs.toy_text.frozen_lake.FrozenLakeEnv
    env = FrozenLakeEnv(desc, map_name, is_slippery)
    return env
