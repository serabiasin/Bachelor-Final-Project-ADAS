import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from .NeuralNetBase import YOLOConv2D


class _ResBlock(Model):
    def __init__(
        self,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None,
    ):
        super(_ResBlock, self).__init__()
        self.conv1 = YOLOConv2D(
            filters=filters_1,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv2 = YOLOConv2D(
            filters=filters_2,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.add = layers.Add()

    def call(self, x):
        ret = self.conv1(x)
        ret = self.conv2(ret)
        x = self.add([x, ret])
        return x


class ResBlock(Model):
    def __init__(
        self,
        filters_1: int,
        filters_2: int,
        iteration: int,
        activation: str = "mish",
        kernel_regularizer=None,
    ):
        super(ResBlock, self).__init__()
        self.iteration = iteration
        self.sequential = Sequential()
        for _ in range(self.iteration):
            self.sequential.add(
                _ResBlock(
                    filters_1=filters_1,
                    filters_2=filters_2,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                )
            )

    def call(self, x):
        return self.sequential(x)


class CSPResNet(Model):
    """
    Cross Stage Partial connections(CSP)
    """

    def __init__(
        self,
        filters_1: int,
        filters_2: int,
        iteration: int,
        activation: str = "mish",
        kernel_regularizer=None,
    ):
        super(CSPResNet, self).__init__()
        self.pre_conv = YOLOConv2D(
            filters=filters_1,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        # Do not change the order of declaration
        self.part2_conv = YOLOConv2D(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.part1_conv1 = YOLOConv2D(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.part1_res_block = ResBlock(
            filters_1=filters_1 // 2,
            filters_2=filters_2,
            iteration=iteration,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.part1_conv2 = YOLOConv2D(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

        self.concat1_2 = layers.Concatenate(axis=-1)

        self.post_conv = YOLOConv2D(
            filters=filters_1,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, x):
        x = self.pre_conv(x)

        part2 = self.part2_conv(x)

        part1 = self.part1_conv1(x)
        part1 = self.part1_res_block(part1)
        part1 = self.part1_conv2(part1)

        x = self.concat1_2([part1, part2])

        x = self.post_conv(x)
        return x


class SPP(Model):
    """
    Spatial Pyramid Pooling layer(SPP)
    """

    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = layers.MaxPooling2D((13, 13), strides=1, padding="same")
        self.pool2 = layers.MaxPooling2D((9, 9), strides=1, padding="same")
        self.pool3 = layers.MaxPooling2D((5, 5), strides=1, padding="same")
        self.concat = layers.Concatenate(axis=-1)

    def call(self, x):
        return self.concat([self.pool1(x), self.pool2(x), self.pool3(x), x])


class CSPDarknet53(Model):
    def __init__(
        self,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer=None,
    ):
        super(CSPDarknet53, self).__init__(name="CSPDarknet53")
        self.conv0 = YOLOConv2D(
            filters=32,
            kernel_size=3,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
        )

        self.res_block1 = CSPResNet(
            filters_1=64,
            filters_2=64,
            iteration=1,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
        )
        self.res_block2 = CSPResNet(
            filters_1=128,
            filters_2=64,
            iteration=2,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
        )
        self.res_block3 = CSPResNet(
            filters_1=256,
            filters_2=128,
            iteration=8,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
        )

        self.res_block4 = CSPResNet(
            filters_1=512,
            filters_2=256,
            iteration=8,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
        )

        self.res_block5 = CSPResNet(
            filters_1=1024,
            filters_2=512,
            iteration=4,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv72 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv73 = YOLOConv2D(
            filters=1024,
            kernel_size=3,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv74 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )

        self.spp = SPP()

        self.conv75 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv76 = YOLOConv2D(
            filters=1024,
            kernel_size=3,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )
        self.conv77 = YOLOConv2D(
            filters=512,
            kernel_size=1,
            activation=activation1,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, x):
        x = self.conv0(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        route1 = x

        x = self.res_block4(x)

        route2 = x

        x = self.res_block5(x)
        x = self.conv72(x)
        x = self.conv73(x)
        x = self.conv74(x)

        x = self.spp(x)

        x = self.conv75(x)
        x = self.conv76(x)
        x = self.conv77(x)

        route3 = x

        return (route1, route2, route3)
