#!/usr/bin/env python3
"""YOLO module"""
import tensorflow.keras as K
import numpy as np


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

    @staticmethod
    def sigmoid(X):
        """
        Computes the sigmoid

        Args:
            X (numpy.ndarray): Contains the input data

        Returns:
            numpy.ndarray: The sigmoid matrix

        """
        return 1 / (1 + np.exp((-1) * X))

    def process_outputs(self, outputs, image_size):
        """
        Processes model outputs

        Args:
            outputs (list(numpy.ndarray)): Is containing the predictions from
                the Darknet model for a single image.
            image_size (numpy.ndarray): Is containing the original size of the
                image.

        Returns:
            tuple(numpy.ndarray): Preprocessed boxes, confidences, class
                probabilities.

        """
        boxes = [output[..., :4] for output in outputs]
        confidences = [output[..., 4:5] for output in outputs]
        classes_probs = [output[..., 5:] for output in outputs]
        bboxes = []

        for idx, box in enumerate(boxes):
            gh, gw, a, _ = box.shape
            cx = np.arange(gw)
            cx = cx.reshape((1, gw, 1))
            cx = np.repeat(cx, gh, axis=0)
            cx = np.repeat(cx, a, axis=2)

            cy = np.arange(gh)
            cy = cy.reshape((gh, 1, 1))
            cy = np.repeat(cy, gw, axis=1)
            cy = np.repeat(cy, a, axis=2)

            aw = self.anchors[idx, ..., 0]
            ah = self.anchors[idx, ..., 1]
            tx = box[..., 0]
            ty = box[..., 1]
            tw = box[..., 2]
            th = box[..., 3]

            bx = (Yolo.sigmoid(tx) + cx) / gh
            by = (Yolo.sigmoid(ty) + cy) / gw
            bw = (np.exp(tw) * aw) / self.model.input_shape[1]
            bh = (np.exp(th) * ah) / self.model.input_shape[2]

            box[..., 0] = (bx - bw / 2) * image_size[1]
            box[..., 1] = (by - bh / 2) * image_size[0]
            box[..., 2] = (bx + bw / 2) * image_size[1]
            box[..., 3] = (by + bh / 2) * image_size[0]
            bboxes.append(box)

        box_confidences = [Yolo.sigmoid(c) for c in confidences]
        box_class_probs = [Yolo.sigmoid(c) for c in classes_probs]
        return (bboxes, box_confidences, box_class_probs)
