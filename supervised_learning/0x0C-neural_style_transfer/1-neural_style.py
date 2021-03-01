#!/usr/bin/env python3
"""Neural Style Transfer module"""
import tensorflow as tf
import numpy as np


class NST:
    """Neural Style Transfer class

    Attributes:
        style_layers (list): The pretrained slected layers for the style.
        content_layers (string): Is represent the pretrained slected layer for
            the content.

    Raises:
        TypeError: If style_image is not np.ndarray with shape (h, w, 3)
        TypeError: If content_image is not np.ndarray with shape (h, w, 3)
        TypeError: If beta is a negative value.
        TypeError: If alpha is a negative value.

    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,  alpha=1e4, beta=1):
        """Initaializer"""
        if not isinstance(style_image, np.ndarray)\
                or len(style_image.shape) != 3\
                or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(content_image, np.ndarray)\
                or len(content_image.shape) != 3\
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()
        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales the image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray): Is containing the image to be scaled of shape
                (h, w, 3) where respectively the height the width and the
                number of channels.

        Returns:
            tf.Tensor: The scaled image

        """
        if not isinstance(image, np.ndarray)\
                or len(image.shape) != 3\
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)

        image = image[tf.newaxis, ...]
        image = tf.image.resize_images(
            image,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """Creates the model used to calculate the cost

        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        s_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        c_output = vgg.get_layer(self.content_layer).output
        outputs = s_outputs + [c_output]
        return tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
