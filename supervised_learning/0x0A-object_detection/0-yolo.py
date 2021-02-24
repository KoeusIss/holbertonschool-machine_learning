#!/usr/bin/env python3
"""YOLO module"""
import tensorflow.keras as K


class Yolo:
    """
    You Only Look Once class

    Attributes:
        model (K.model): The Darknet keras model
        class_names (list): The list of COCO dataset classe names
        class_t (float): The box score threshold.
        nms_t (float): The IoU threshold for non-max suppression
        anchors (numpy.ndarray): The anchor boxes

    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializer
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().rstrip('\n').split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
