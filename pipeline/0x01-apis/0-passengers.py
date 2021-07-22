#!/usr/bin/env python3
"""API module"""
import requests


def availableShips(passengerCount):
    """Returns the list of ships that can hold a given number of passenger

    Arguments:
        passengerCount {int} -- The number of passengers

    Returns:
        set -- Contains the name of ships.
    """
    entry_point = "https://swapi-api.hbtn.io/api/starships/"
    response = requests.get(entry_point)
    content = response.json()
    ships = set()

    while content['next']:
        page_result = content['results']
        for ship in page_result:
            passengers = ship['passengers']
            try:
                passengers = ''.join(passengers.split(','))
                passengers = int(passengers)
            except ValueError:
                passengers = 0
            if passengers >= passengerCount:
                ships.add(ship['name'])

        response = requests.get(content['next'])
        content = response.json()

    return ships
