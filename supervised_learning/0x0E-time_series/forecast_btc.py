#!/usr/bin/env python3
"""Time Series Forecasting module"""

# Dependencies
# -----------------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib as mpl
import matplotlib.pyplot as plt

preprocess_data = __import__('preprocess_data').preprocess_data

# Universal constant
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
INUPT_WIDTH = 24
LABEL_WIDTH = 1
STRIDE_WIDTH = 1
BUFFER_SIZE = 10000
CSV_PATH = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'

# Preprocessing data
# -----------------------------------------------------------------------------
train_ds, valid_ds = preprocess_data(
    CSV_PATH,
    INUPT_WIDTH,
    LABEL_WIDTH,
    BUFFER_SIZE,
    BATCH_SIZE,
    STRIDE_WIDTH
    )

# Building model
# -----------------------------------------------------------------------------
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])

# Compile and fit model
# -----------------------------------------------------------------------------
lstm_model.compile(
    optimizer='adam',
    loss='mse'
)
history = lstm_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=800,
    validation_data=valid_ds,
    validation_steps=80
)
