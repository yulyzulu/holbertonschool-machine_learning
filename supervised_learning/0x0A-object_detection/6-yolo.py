#!/usr/bin/env python3
""" Yolo Class """

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max suppression method- Return a tuple of
           (box_predictions, predicted_box_classes, predicted_box_scores)"""
        index = np.lexsort((-box_scores, box_classes))
        box_predictions = filtered_boxes[index]
        predict_box_classes = box_classes[index]
        predict_box_scores = box_scores[index]
        _, number_counts = np.lib.arraysetops.unique(predict_box_classes,
                                                     return_counts=True)
        i = 0
        acummulated = 0
        for number_count in number_counts:
            print(number_count)
            while i < acummulated + number_count:
                j = i + 1
                while j < acummulated + number_count:
                    b_p1 = box_predictions[i]
                    b_p2 = box_predictions[j]
                    x_x1 = np.maximum(b_p1[0], b_p2[0])
                    y_y1 = np.maximum(b_p1[1], b_p2[1])
                    x_x2 = np.minimum(b_p1[2], b_p2[2])
                    y_y2 = np.minimum(b_p1[3], b_p2[3])
                    inter_area = (y_y2 - y_y1) * (x_x2 - x_x1)
                    box1_area = (b_p1[3] - b_p1[1])*(b_p1[2] - b_p1[0])
                    box2_area = (b_p2[3] - b_p2[1])*(b_p2[2] - b_p2[0])
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area/union_area
                    if iou > self.nms_t:
                        box_predictions = np.delete(box_predictions,
                                                    j, axis=0)
                        predict_box_scores = np.delete(predict_box_scores,
                                                       j, axis=0)
                        predict_box_classes = np.delete(predict_box_classes,
                                                        j, axis=0)
                        number_count -= 1
                    else:
                        j += 1
                i += 1
            acummulated += number_count
        return box_predictions, predict_box_classes, predict_box_scores

    @staticmethod
    def load_images(folder_path):
        """Static method to load image"""
        images_path = glob.glob(folder_path + "/*")
        images = [cv2.imread(i) for i in images_path]
        return images, images_path

    def preprocess_images(self, images):
        """Preprocess method"""
        new_images = []
        images_shapes = []
        for image in images:
            images_shapes.append([image.shape[0], image.shape[1]])
            size_image = (self.model.input.shape[1], self.model.input.shape[2])
            img = (cv2.resize(image, size_image,
                              interpolation=cv2.INTER_CUBIC)) / 255
            new_images.append(img)
        return (np.array(new_images), np.array(images_shapes))

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Show boxes method that displays the image with all boundary
            boxes, class names, and box scores"""
        directory = 'detections'
        for i in range(len(boxes)):
            classes_name = str(self.class_names[box_classes[i]])
            box_score = str(round(box_scores[i], 2))
            label = classes_name + " " + box_score
            color = (255, 0, 0)
            red = (0, 0, 255)
            x = int(boxes[i][0])
            y = int(boxes[i][1])
            x_w = int(boxes[i][2])
            y_h = int(boxes[i][3])

            cv2.rectangle(image, (x, y), (x_w, y_h), color, 2)
            cv2.putText(image, label, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, red, 1, cv2.LINE_AA)
        cv2.imshow(file_name, image)
        k = cv2.waitKey(0)
        if not os.path.exists(directory):
            os.mkdir(directory)
        if k == ord('s'):
            cv2.imwrite(os.path.join(directory, file_name), image)
        cv2.destroyAllWindows()
