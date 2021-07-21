#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.pop("Weighted_Price")
df['Date'] = pd.to_datetime(df.pop('Timestamp'), unit='s')
df = df.set_index('Date')
df[['Close']] = df[['Close']].ffill()
df['High'] = df['High'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df.fillna({"Volume_(BTC)": 0, "Volume_(Currency)": 0}, inplace=True)

df = df.loc['2017-01-01 00:00:00':]

high = df['High'].groupby(pd.Grouper(freq='D')).max()
low = df['Low'].groupby(pd.Grouper(freq='D')).min()
open = df['Open'].groupby(pd.Grouper(freq='D')).mean()
close = df['Close'].groupby(pd.Grouper(freq='D')).mean()
volume_btc = df['Volume_(BTC)'].groupby(pd.Grouper(freq='D')).sum()
volume_currency = df['Volume_(Currency)'].groupby(pd.Grouper(freq='D')).sum()

plt.figure(figsize=(16, 8))
plt.plot(open, label='Open')
plt.plot(high, label='High')
plt.plot(low, label='Low')
plt.plot(close, label='Close')
plt.plot(volume_btc, label='Volume_(BTC)')
plt.plot(volume_currency, label='Volume_(Currency)')
plt.legend()
plt.show()
