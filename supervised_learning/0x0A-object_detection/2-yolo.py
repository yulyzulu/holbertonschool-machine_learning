#!/usr/bin/env python3
""" Yolo Class """

import tensorflow as tf
import tensorflow.keras as K
import numpy as np


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

    def sigmoid(self, x):
        """Sigmoid"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """ Process_outpus method- Returns a tuple of
          (boxes, box_confidences, box_class_probs)"""
        boxes = []
        box_class = []
        box_confidences = []
        i = 0
        for output in outputs:
            boxes.append(output[:, :, :, 0:4])
            box_class.append(self.sigmoid(output[:, :, :, 5:]))
            box_confidences.append(self.sigmoid(output[:, :, :, 4:5]))

        for box in boxes:
            H_box = box.shape[0]
            W_box = box.shape[1]
            anchor_box = box.shape[2]

            the_box = np.zeros((H_box, W_box, anchor_box))

            ind_x = np.arange(W_box)
            ind_y = np.arange(H_box)
            ind_x = ind_x.reshape(1, W_box, 1)
            ind_y = ind_y.reshape(H_box, 1, 1)

            box_x = the_box + ind_x
            box_y = the_box + ind_y

            tx = box[..., 0]
            ty = box[..., 1]
            tw = box[..., 2]
            th = box[..., 3]

            sig_tx = self.sigmoid(tx)
            sig_ty = self.sigmoid(ty)

            bx = sig_tx + box_x
            by = sig_ty + box_y
            bx = bx / W_box
            by = by / H_box

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            inp_w = self.model.input.shape[1].value
            inp_h = self.model.input.shape[2].value

            bw = bw / inp_w
            bh = bh / inp_h

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            box[..., 0] = x1 * image_size[1]
            box[..., 1] = y1 * image_size[0]
            box[..., 2] = x2 * image_size[1]
            box[..., 3] = y2 * image_size[0]
            i = i + 1

        return (boxes, box_confidences, box_class)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter_boxes method- Returns a tuple of
           (filtered_boxes, box_classes, box_scores)"""
        fil_scores = []
        fil_classes = []
        fil_boxes = []
        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            box_class_scores = np.max(box_score, axis=-1).reshape(-1)
            box_classes_del = np.where(box_class_scores < self.class_t)
            filter_mask = box_class_scores >= self.class_t

            scores = filter_mask * box_class_scores
            scores = scores[scores > 0]
            classes = np.argmax(box_score, axis=-1).reshape(-1)
            classes = np.delete(classes, box_classes_del)
            a, b, c, _ = boxes[i].shape
            fil_mask_reshape = filter_mask.reshape(a, b, c, 1)
            boxes_list = boxes[i] * fil_mask_reshape
            boxes_list = boxes_list[boxes_list != 0]
            fil_scores.append(scores)
            fil_classes.append(classes)
            fil_boxes.append(boxes_list.reshape(-1, 4))
        fil_scores = np.concatenate(fil_scores)
        fil_boxes = np.concatenate(fil_boxes)
        fil_classes = np.concatenate(fil_classes)
        return (fil_boxes, fil_classes, fil_scores)
