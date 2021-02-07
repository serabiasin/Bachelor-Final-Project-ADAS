from tensorflow.keras import backend, layers, optimizers
import tensorflow as tf

import numpy as np
from typing import Union
from .YOLOEngine import YOLOVision
import cv2

class ADASEngine:
    def __init__(self,cameraDir=None):
        self.anchors = [
            [[12, 16], [19, 36], [40, 28]],
            [[36, 75], [76, 55], [72, 146]],
            [[142, 110], [192, 243], [459, 401]],
        ]


        self.classes = None
        self.input_size = None
        self.weights_path=None

        self.strides = np.array([8, 16, 32])
        self.xyscales = [1.2, 1.1, 1.05]
        self.batch_size = 32
        self.input_size = 608
        self._has_weights = False
        
        self.model = None

    def build_model(
        self,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer = tf.keras.regularizers.l2(0.0005),
        ):
        # pylint: disable=missing-function-docstring
        self._has_weights=False
        backend.clear_session()

        # height, width, channels
        inputs=layers.Input([self.input_size[1], self.input_size[0], 3])
        self.model = YOLOVision(
                anchors=self.anchors,
                num_classes=len(self.classes),
                xyscales=self.xyscales,
                activation0=activation0,
                activation1=activation1,
                kernel_regularizer=kernel_regularizer,
            )
        self.model(inputs)

    def load_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("yolov4.weights", weights_type="yolo")
            yolo.load_weights("checkpoints")
        """
        self.model.load_weights(weights_path)

        self._has_weights = True

    def check_model(self):
        self.model.summary()
