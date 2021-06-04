from keras.layers import Conv2D, UpSampling2D, LeakyReLU, SeparableConv2D, BatchNormalization, Activation, add
from keras.models import Model
from keras.applications.mobilenet import MobileNet


class SDWConv(model):


    def __init__(self, filters, kernel):
        super(SDWConv, self).__init__()
        self.separableconv = SeparableConv2D(
            filtres, kernel, padding='same')
        self.batchnorm=BatchNormalization()
        self.relu=LeakyReLU()

    def call(self,x):
        pass
