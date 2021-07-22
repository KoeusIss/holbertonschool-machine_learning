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

    planets = set()
    while content['next']:
        page_result = content['results']

        for sentinent in page_result:
            planet_url = sentinent['homeworld']
            if planet_url:
                name = requests.get(planet_url).json()['name']
                planets.add(name)

        response = requests.get(content['next'])
        content = response.json()
    return planets


planets = sentientPlanets()
for planet in planets:
    print(planet)
