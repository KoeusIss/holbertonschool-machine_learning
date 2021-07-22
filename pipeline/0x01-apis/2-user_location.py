#!/usr/bin/env python3
"""API module"""
import requests
from datetime import datetime
import sys

if __name__ == '__main__':
    url = sys.argv[1]

    response = requests.get(url)

    if response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        limit = response.headers['X-Ratelimit-Reset']
        now = datetime.now().timestamp()
        difference = (int(limit) - int(now)) / 60
        print('Reset in {} min'.format(int(difference)))
    elif response.status_code == 200:
        print(response.json()['location'])
