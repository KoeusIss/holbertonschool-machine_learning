#!/usr/bin/env python3
"""Transfer Learning module"""
import tensorflow as tf
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
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return X_p, Y_p


if __name__ == "__main__":

    (X_train, Y_train), _ = K.datasets.cifar10.load_data()
    batch_size=100
    epochs=20

    base_model = K.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    X, Y = preprocess_data(X_train, Y_train)

    base_model.trainable = False

    source_model = K.Sequential()
    source_model.add(K.layers.Lambda(
        lambda x: K.backend.resize_images(x, 7, 7, 'channels_last'),
        input_shape=(32, 32, 3),
        trainable=False
    ))
    source_model.add(base_model)


    X_feature = source_model.predict(X)

    feature_model = K.Sequential()
    feature_model.add(K.layers.BatchNormalization())
    feature_model.add(K.layers.Dense(512, activation='relu'))
    feature_model.add(K.layers.Dropout(0.2))
    feature_model.add(K.layers.BatchNormalization())
    feature_model.add(K.layers.Dense(256, activation='relu'))
    feature_model.add(K.layers.Dropout(0.2))
    feature_model.add(K.layers.Dense(10, activation='softmax'))

    datagen = K.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizental_flip=True
    )
    generator = datagen.flow(
        X,
        Y,
        batch_size=25
    )

    feature_model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    steps = X.shape[0] // batch_size
    feature_model.fit(
        generator,
        steps_per_epoch=steps,
        epochs=epochs,
    )
    feature_model.save('cifar10.h5')
