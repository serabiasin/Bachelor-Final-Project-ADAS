import colorsys
import os

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from utils_object.utils import BBoxUtility, letterbox_image, ssd_correct_boxes
from config_kittipascal import config


class detector(object):

    def __init__(self, weight_path=None, network=None):
        self.classes = config.CLASSES
        self.input_shape = config.IMAGE_SIZE_300
        self._load_weigth(weight_path=weight_path, network=network)
        self.bbox_util = BBoxUtility(len(self.classes))

    def _load_weigth(self, weight_path=None, network=None, groups=3):

        weight_path = os.path.expanduser(weight_path)
        assert weight_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        if network == 'mobilenetv1':
            from models.ssd_mobilenetv1_300 import SSD300
        elif network == 'mobilenetv1_old':
            from models.ssd_mobilenetv1_300_old import SSD300
        elif network == 'mobilenetv2':
            from models.ssd_mobilenetv2_300 import SSD300
        elif network == 'mobilenetv3':
            from models.ssd_mobilenetv3_300 import SSD300
        elif network == 'efficientnet':
            from models.ssd_efficientnetb0 import SSD300
        elif network == 'experiment':
            from models.ssd_experiment import SSD300
        else:
            raise ValueError(
                "You need network choice `mobilenetv1`, `mobilenetv2`, `mobilenetv3`.")

        self.model = SSD300(config.IMAGE_SIZE_300, len(
            self.classes), anchors=config.ANCHORS_SIZE_300)
        self.model.load_weights(weight_path)

        # Set every class' color
        hsv_tuples = [(x / len(self.classes), 1., 1.)
                      for x in range(len(self.classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    @tf.function
    def get_pred(self, photo):
        preds = self.model(photo, training=False)
        return preds

    # Detected Image
    def detect_image(self, image):

        image_shape = np.array(np.shape(image)[0:2])
        crop_image, x_offset, y_offset = letterbox_image(
            image, (self.input_shape[0], self.input_shape[1]))
        photo = np.array(crop_image, dtype=np.float64)

        # Normalization
        photo = preprocess_input(np.reshape(
            photo, [1, self.input_shape[0], self.input_shape[1], 3]))
        preds = self.get_pred(photo).numpy()

        # Decode
        results = self.bbox_util.detection_out(
            preds, confidence_threshold=config.CONFIDENCE)

        if len(results[0]) <= 0:
            return image

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:,
                                                            2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(
            det_conf) if conf >= config.CONFIDENCE]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = np.expand_dims(det_xmin[top_indices], axis=-1)
        top_ymin = np.expand_dims(det_ymin[top_indices], axis=-1)
        top_xmax = np.expand_dims(det_xmax[top_indices], axis=-1)
        top_ymax = np.expand_dims(det_ymax[top_indices], axis=-1)

        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            (self.input_shape[0], self.input_shape[1])), image_shape)

        boxes_depth = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            (self.input_shape[0], self.input_shape[1])), np.array(240, 320))

        font = ImageFont.truetype(
            font='./font/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] +
                     np.shape(image)[1]) // self.input_shape[0]

        for i, c in enumerate(top_label_indices):

            predicted_class = self.classes[int(c) - 1]
            score = top_conf[i]

            ymin, xmin, ymax, xmax = boxes[i]
            ymin = ymin - 5
            xmin = xmin - 5
            ymax = ymax - 5
            xmax = xmax - 5

            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymax = min(np.shape(image)[0], np.floor(
                ymax + 0.5).astype('int32'))
            xmax = min(np.shape(image)[1], np.floor(
                xmax + 0.5).astype('int32'))

            center_y = (ymax+ymin)/2
            center_x = (xmax+xmin)/2
            
            # draw Bounding box
            label = "{}:{:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            if ymin - label_size[1] >= 0:
                text_origin = np.array((xmin, ymin - label_size[1]))
            else:
                text_origin = np.array((xmin, ymin + 1))
            for i in range(thickness):

              draw.rectangle(
                  [xmin + i, ymin + i, xmax - i, ymax - i],
                  outline=self.colors[int(c)-1])
              leftUpPoint = (center_x-10, center_y-10)
              rightDownPoint = (center_x+10, center_y+10)
              twoPointList = [leftUpPoint, rightDownPoint]
              draw.ellipse(twoPointList, fill=(255, 0, 0, 255))
              draw.rectangle(
                  [tuple(text_origin), tuple(text_origin + label_size)],
                  fill=self.colors[int(c)-1])
              draw.text(text_origin, str(label, 'UTF-8'),
                        fill='yellow', font=font)

            del draw
        return image

    def detected_bbox(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        crop_image, x_offset, y_offset = letterbox_image(
            image, (self.input_shape[0], self.input_shape[1]))
        photo = np.array(crop_image, dtype=np.float64)

        # Normalization
        photo = preprocess_input(np.reshape(
            photo, [1, self.input_shape[0], self.input_shape[1], 3]))
        preds = self.get_pred(photo).numpy()

        # Decode
        results = self.bbox_util.detection_out(
            preds, confidence_threshold=config.CONFIDENCE)

        if len(results[0]) <= 0:
            return []

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:,
                                                            2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(
            det_conf) if conf >= config.CONFIDENCE]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = np.expand_dims(det_xmin[top_indices], axis=-1)
        top_ymin = np.expand_dims(det_ymin[top_indices], axis=-1)
        top_xmax = np.expand_dims(det_xmax[top_indices], axis=-1)
        top_ymax = np.expand_dims(det_ymax[top_indices], axis=-1)

        #image shape -> original file size
        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            (self.input_shape[0], self.input_shape[1])), image_shape)
        boxes_depth = ssd_correct_boxes(
            top_ymin, top_xmin, top_ymax, top_xmax, np.array(
                (self.input_shape[0], self.input_shape[1])),  np.array((240, 320)))

        predict_results = []

        for i, c in enumerate(top_label_indices):

            predicted_class = int(c) - 1
            score = top_conf[i]

            ymin, xmin, ymax, xmax = boxes[i]

            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymax = min(np.shape(image)[0], np.floor(
                ymax + 0.5).astype('int32'))
            xmax = min(np.shape(image)[1], np.floor(
                xmax + 0.5).astype('int32'))

            predict_results.append([xmin, ymin, xmax, ymax, predicted_class])

        return predict_results

    def combine_it(self, image, depth_matrix):
        image_shape = np.array(np.shape(image)[0:2])
        crop_image, x_offset, y_offset = letterbox_image(
            image, (self.input_shape[0], self.input_shape[1]))
        photo = np.array(crop_image, dtype=np.float64)

        # Normalization
        photo = preprocess_input(np.reshape(
            photo, [1, self.input_shape[0], self.input_shape[1], 3]))
        preds = self.get_pred(photo).numpy()

        # Decode
        results = self.bbox_util.detection_out(
            preds, confidence_threshold=config.CONFIDENCE)

        if len(results[0]) <= 0:
            return image

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:,
                                                            2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(
            det_conf) if conf >= config.CONFIDENCE]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = np.expand_dims(det_xmin[top_indices], axis=-1)
        top_ymin = np.expand_dims(det_ymin[top_indices], axis=-1)
        top_xmax = np.expand_dims(det_xmax[top_indices], axis=-1)
        top_ymax = np.expand_dims(det_ymax[top_indices], axis=-1)

        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            (self.input_shape[0], self.input_shape[1])), image_shape)

        boxes_depth = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            (self.input_shape[0], self.input_shape[1])), np.array((240, 320)))

        font = ImageFont.truetype(
            font='./font/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] +
                     np.shape(image)[1]) // self.input_shape[0]

        for i, c in enumerate(top_label_indices):

            predicted_class = self.classes[int(c) - 1]
            score = top_conf[i]

            ymin, xmin, ymax, xmax = boxes[i]
            ymin = ymin - 5
            xmin = xmin - 5
            ymax = ymax - 5
            xmax = xmax - 5

            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymax = min(np.shape(image)[0], np.floor(
                ymax + 0.5).astype('int32'))
            xmax = min(np.shape(image)[1], np.floor(
                xmax + 0.5).astype('int32'))

            ymin_d, xmin_d, ymax_d, xmax_d = boxes_depth[i]
            ymin_d = max(0, np.floor(ymin_d + 0.5).astype('int32'))
            xmin_d = max(0, np.floor(xmin_d + 0.5).astype('int32'))
            ymax_d = min(np.shape(image)[0], np.floor(
                ymax_d + 0.5).astype('int32'))
            xmax_d = min(np.shape(image)[1], np.floor(
                xmax_d + 0.5).astype('int32'))

            center_y = int((ymax_d+ymin_d)/2)
            center_x = int((xmax_d+xmin_d)/2)

            # draw Bounding box
            label = "{}:{:.2f}".format(
                predicted_class, depth_matrix[0,  center_y,center_x, 0])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            if ymin - label_size[1] >= 0:
                text_origin = np.array((xmin, ymin - label_size[1]))
            else:
                text_origin = np.array((xmin, ymin + 1))
            for i in range(thickness):

              draw.rectangle(
                  [xmin + i, ymin + i, xmax - i, ymax - i],
                  outline=self.colors[int(c)-1])
              draw.rectangle(
                  [tuple(text_origin), tuple(text_origin + label_size)],
                  fill=self.colors[int(c)-1])
              draw.text(text_origin, str(label, 'UTF-8'),
                        fill='yellow', font=font)

            del draw
        return image
