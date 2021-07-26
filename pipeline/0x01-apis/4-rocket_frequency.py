#!/usr/bin/env python3
"""API module"""
import requests
from datetime import datetime


if __name__ == '__main__':
    base_url = "https://api.spacexdata.com/v3"

    response = requests.get(base_url + "/rockets")
    content = response.json()

    rockets = []

    for rocket in content:
        rockets.append(rocket['rocket_name'])

    launches = dict()
    for rocket in rockets:
        payload = {"rocket_name": rocket}
        response = requests.get(base_url + "/launches", params=payload)
        content = response.json()
        launches['rocket'] = len(content)

        print("{}: {}".format(rocket, len(content)))
