#!/usr/bin/env python3
"""Hyperparameter Tuning"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization

# Importing dataset
from tensorflow.keras.datasets import mnist

# Getting featurees and labels
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Flatten data
m, h, w = X_train.shape
mv, _, _ = X_test.shape
X_train = X_train.reshape(m, h * w)
X_test = X_test.reshape(mv, h * w)

# Data Normalization
X_train = X_train / 255
X_test = X_test / 255

# Labels Encoding
Y_train_oh = K.utils.to_categorical(Y_train)
Y_test_oh = K.utils.to_categorical(Y_test)


def get_model(lr, beta1, beta2, nodes):
    """Creates a model base on the provided parameters

    Arguments:
        lr {float} -- Is the learning rate for Adam optimizer
        beta1 {float} -- Is the exponential decay rate
        beta2 {float} -- Is the exponential decay rate
        nodes {tuple} -- Contains the number of units in each layer

    Returns:
        keras.model -- A keras model
    """
    inputs = K.Input(
        shape=(784,),
        name="inputs_layer"
    )
    layer = K.layers.Dense(
        nodes[0],
        activation="relu",
        name="first_layer"
    )(inputs)
    layer = K.layers.Dense(
        nodes[1],
        activation="relu",
        name="second_layer"
    )(layer)
    layer = K.layers.Dense(
        nodes[2],
        activation="relu",
        name="third_layer"
    )(layer)
    outputs = K.layers.Dense(
        10,
        activation="softmax",
        name="last_layer"
    )(layer)
    model = K.Model(inputs, outputs)

    optimizer = K.optimizers.Adam(
        lr=lr,
        beta_1=beta1,
        beta_2=beta2
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def fit_model(hyperparams):
    """Fit the model

    Arguments:
        hyperparams {list} -- Contains the passed parameters

    Returns:
        float -- The loss
    """
    lr = hyperparams[0][0]
    beta1 = hyperparams[0][1]
    beta2 = hyperparams[0][2]
    batch_size = int(hyperparams[0][3])
    epochs = int(hyperparams[0][4])
    nodes = (
        int(hyperparams[0][5]),
        int(hyperparams[0][6]), 
        int(hyperparams[0][7])
    )

    model = get_model(lr, beta1, beta2, nodes)

    callback = []

    early = K.callbacks.EarlyStopping(monitor='loss', patience=3)
    callback.append(early)

    filepath = "lr_{}-beta1_{}-beta2_{}-batchsize_{}-epochs_{}-nodes_{}.{}.{}".format(
        lr.round(4),
        beta1.round(4),
        beta2.round(4),
        batch_size,
        epochs,
        nodes[0],
        nodes[1],
        nodes[2]
    )
    best = K.callbacks.ModelCheckpoint(
        filepath,
        save_best_only=True,
        monitor='loss'
    )
    callback.append(best)

    blackbox = model.fit(
        X_train,
        Y_train_oh,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        verbose=0,
        callbacks=callback
    )
    loss = blackbox.history['val_loss'][-1]
    return loss


# Space domain
lr_domain = (0.00001, 0.1)
beta1_domain = (0.9, 0.9999)
beta2_domain = (0.9, 0.9999)
batch_domain = [25, 50, 75, 100, 125, 150, 200]
epochs_domain = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
first_domain = [16, 32, 64, 128, 256, 512]
second_domain = [16, 32, 64, 128, 256, 512]
third_domain = [16, 32, 64, 128, 256, 512]

# Space bounds
bounds = [
    {"name": "lr", "type": "continuous", "domain": lr_domain},
    {"name": "beta1", "type": "continuous", "domain": beta1_domain},
    {"name": "beta2", "type": "continuous", "domain": beta2_domain},
    {"name": "batch_size", "type": "discrete", "domain": batch_domain},
    {"name": "epochs", "type": "discrete", "domain": epochs_domain},
    {"name": "first", "type": "discrete", "domain": first_domain},
    {"name": "second", "type": "discrete", "domain": second_domain},
    {"name": "third", "type": "discrete", "domain": third_domain},
]
# Initialize Bayesian optimization
b_optimization = BayesianOptimization(
    fit_model,
    domain=bounds,
    model_type="GP",
    initial_design_numdata=1,
    acquisition_type="EI",
    maximize=True,
    verbosity=True
)
# Run optimization
b_optimization.run_optimization(max_iter=30)

# Save report
b_optimization.save_report('bayes_opt.txt')

# Print Convergence
b_optimization.plot_convergence()
