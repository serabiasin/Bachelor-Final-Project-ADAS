from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
import tensorflow


class UpscaleBlock(Model):
    def __init__(self, filters, name):
        super(UpscaleBlock, self).__init__()
        self.up = UpSampling2D(
            size=(2, 2), interpolation='bilinear', name=name+'_upsampling2d')
        self.concat = Concatenate(name=name+'_concat')  # Skip connection
        self.convA = Conv2D(filters=filters, kernel_size=3,
                            strides=1, padding='same', name=name+'_convA')
        self.reluA = LeakyReLU(alpha=0.4)
        self.convB = Conv2D(filters=filters, kernel_size=3,
                            strides=1, padding='same', name=name+'_convB')
        self.reluB = LeakyReLU(alpha=0.4)

    def call(self, x):
        b = self.reluB(self.convB(self.reluA(
            self.convA(self.concat([self.up(x[0]), x[1]])))))
        return b


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.base_model = MobileNetV2(input_shape=(
            None, None, 3), include_top=False, weights='imagenet')
        print('Base model loaded {}'.format(MobileNetV2.__name__))

        # Create encoder model that produce final features along with multiple intermediate features
        outputs = [self.base_model.outputs[-1]]
        #New
        upsample_layer = ['block_12_depthwise', 'block_12_project',
                          'block_5_depthwise', 'block_5_project',
                          'block_2_depthwise', 'block_2_project',
                          'Conv1_relu']
        for name in upsample_layer:

            outputs.append(self.base_model.get_layer(name).output)

        self.encoder = Model(inputs=self.base_model.inputs, outputs=outputs)

    def call(self, x):
        return self.encoder(x)


class Decoder(Model):
    def __init__(self, decode_filters):
        super(Decoder, self).__init__()
        self.conv2 = Conv2D(filters=decode_filters,
                            kernel_size=1, padding='same', name='conv2')
        self.up1_1 = UpscaleBlock(filters=decode_filters//2,  name='up1')
        self.up1_2 = UpscaleBlock(filters=decode_filters//2,  name='up1')

        self.up2_1 = UpscaleBlock(filters=decode_filters//4,  name='up2')
        self.up2_2 = UpscaleBlock(filters=decode_filters//4,  name='up2')

        self.up3_1 = UpscaleBlock(filters=decode_filters//8,  name='up3')
        self.up3_2 = UpscaleBlock(filters=decode_filters//8,  name='up3')

        self.up4 = UpscaleBlock(filters=decode_filters//16, name='up4')
        self.conv3 = Conv2D(filters=1, kernel_size=3,
                            strides=1, padding='same', name='conv3')

    def call(self, features):
        x, expand12, project12, expand5, project5, expand10, project10, conv1 = features[0], features[
            1], features[2], features[3], features[4], features[5], features[6], features[7]
        up0 = self.conv2(x)
        # print(up0)
        up1_1 = self.up1_1([up0, expand12])
        up1_2 = self.up1_2([up0, project12])
        up1_3 = tensorflow.keras.layers.Add()([up1_1, up1_2])
        # print(up1_3)
        # print(up1_2)
        up2_1 = self.up2_1([up1_3, expand5])
        up2_2 = self.up2_2([up1_3, project5])
        up2_3 = tensorflow.keras.layers.Add()([up2_1, up2_2])
        # print(up2)
        up3_1 = self.up3_1([up2_3, expand10])
        up3_2 = self.up3_2([up2_3, project10])
        up3_3 = tensorflow.keras.layers.Add()([up3_1, up3_2])
        # print(up2_3)
        up4 = self.up4([up3_3, conv1])
        return self.conv3(up4)


class DepthEstimate(Model):
    def __init__(self):
        super(DepthEstimate, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(decode_filters=int(
            self.encoder.layers[-1].output[0].shape[-1] // 2))
        print('\nModel created.')

    def call(self, x):
        return self.decoder(self.encoder(x))
