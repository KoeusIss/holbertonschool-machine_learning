#!/usr/bin/env python3
"""API module"""
import requests


def sentientPlanets():
    """Returns the ist of planets name of all sentinent species

    Returns:
        set -- All planets name.
    """
    entry_point = "https://swapi-api.hbtn.io/api/species/"
    response = requests.get(entry_point)
    content = response.json()

    planets = list()
    while True:
        results = content['results']
        for sentinent in results:
            planet_url = sentinent['homeworld']
            if planet_url:
                name = requests.get(planet_url).json()['name']
                planets.append(name)
        if content['next'] is None:
            break
        response = requests.get(content['next'])
        content = response.json()

    return planets
