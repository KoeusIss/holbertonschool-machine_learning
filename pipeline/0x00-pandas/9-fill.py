#!/usr/bin/env python3
"""Pandas module"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.pop("Weighted_Price")
df.fillna({"Volume_(BTC)": 0, "Volume_(Currency)": 0}, inplace=True)
df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()

print(df.head())
print(df.tail())
