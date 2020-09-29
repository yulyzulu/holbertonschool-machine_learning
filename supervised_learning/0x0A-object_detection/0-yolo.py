#!/usr/bin/env python3
""" Yolo Class """

import tensorflow as tf
import tensorflow.keras as K


class Yolo():
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)
        with open(classes_path) as f:
            self.class_names = []
            for i in f:
                i = i.split("\n")
                self.class_names.append(i[0])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
