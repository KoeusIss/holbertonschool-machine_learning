#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves model's configuration in JSON format

    Args:
        network (keras.model): The model to be saved.
        filename (str): The path where should be saved

    Returns:
        None

    """
    json = network.to_json()
    with open(filename, "w") as f:
        f.write(json)
    return None


def load_config(filename):
    """Loads a model withe a specefic configuration

    Args:
        filename (str): The configuration saved path

    Returns:
        keras.model: The model with the saved configuration

    """
    with open(filename, "r") as f:
        json_string = f.read()
    return K.models.model_from_json(json_string)
