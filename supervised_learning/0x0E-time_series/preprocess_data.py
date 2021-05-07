#!/usr/bin/env python3
"""Time Series Forecasting module"""

# Dependencies
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib as mpl
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt


def preprocess_data(
    csv_path,
    input_width,
    label_width=1,
    buffer_size=10000,
    batch_size=32,
    stride=1
):
    # Load from a csv file using panda
    dataframe = pd.read_csv(csv_path)
    # Forward Filling missing data
    df = dataframe.ffill()
    # Drop and parse datetime
    date_time = pd.to_datetime(df.pop('Timestamp'), unit='s')
    # Window the dataframe per hour
    df = df[8::60]
    date_time = date_time[8::60]
    # Drop the highly correlated features
    df.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
    # Split data
    n = len(df)
    train_df = df[:int(n * 0.8)]
    valid_df = df[int(n * 0.8):]

    # Data Normalization
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    valid_df = (valid_df - train_mean) / train_std

    # Split to window
    size = len(train_df)
    window_size = input_width + label_width

    x = []
    y = []
    for i in range(0, size - window_size, stride):
        x.append(train_df.values[i:i + input_width])
        y.append(train_df.values[i + input_width:i + window_size])

    x_train = np.array(x)
    y_train = np.array(y)

    x = []
    y = []
    for i in range(0, size - window_size, stride):
        x.append(valid_df.values[i:i + input_width])
        y.append(valid_df.values[i + input_width:i + window_size])

    x_valid = np.array(x)
    y_valid = np.array(y)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).\
        shuffle(buffer_size).\
        batch(batch_size).\
        repeat()
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).\
        shuffle(buffer_size).\
        batch(batch_size).\
        repeat()

    return train_ds, valid_ds
