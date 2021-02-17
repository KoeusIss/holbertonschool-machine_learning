#!/usr/bin/env python3
"""Transfer Learning module"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """Pre-Processes the data for a model

    Args:
        X (numpy.ndarray): Is containing the CIFAR10 data with shape
            (m, 32, 32, 3) where m is the number of data point, followed by
            height, weight and number of channels.
        Y (numpy.ndarray): Is containing the CIFAR10 labels for x with shape
            (m,) where m is the number of data points.

    Returns:
        numpy.ndarray: X_p is containing the preprocessed X
        numpy.ndarray: Y_p is containing the preprocessed Y

    """
    X_p = X / 255
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p


if __name__ == "__main__":

    (X_train, _), (Y_train, _) = K.datasets.cifar10.load_data()

    vgg = K.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    X, Y = preprocess_data(X_train, Y_train)

    vgg.trainable = False

    model = K.Sequential()
    model.add(K.layers.Lambda(
        lambda x: K.backend.resize_images(x, 7, 7, 'channels_last'),
        input_shape=(32, 32, 3),
        trainable=False
    ))
    model.add(base_model)
    model.add(K.layers.Flatten(trainable=False))

    X_Features = model.predict(X)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.save('cifar10.h5')
