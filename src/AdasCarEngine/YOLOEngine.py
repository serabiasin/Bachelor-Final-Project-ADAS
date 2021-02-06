from tensorflow.keras import Model

from .Backbone import CSPDarknet53
from .Head import YOLOHead
from .Neck import PANet


class YOLOEngine(Model):
    def __init__(self,
                 anchors,
                 num_classes: int,
                 xyscales,
                 activation0: str = "mish",
                 activation1: str = "leaky",
                 kernel_regularizer=None,
                 ):

        super(YOLOEngine, self).__init__()
        self.csp_darknet53=CSPDarknet53(
            activation0=activation0,
            activation1=activation1,
            kernel_regularizer=kernel_regularizer
        )

        self.panet=PANet(
            num_classes=num_classes,
            activation=activation1,
            kernel_regularizer=kernel_regularizer
        )
        self.yolo_head=YOLOHead(
            anchors=anchors, num_class=num_classes,xyscale=xyscales
        )

    def call(self,x):
        x = self.csp_darknet53(x)
        x = self.panet(x)
        x = self.yolov3_head(x)
        return x
