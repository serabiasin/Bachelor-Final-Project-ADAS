from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Reshape, ZeroPadding2D, Concatenate, Input, Activation, Flatten, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, add, GlobalAveragePooling2D, Dense, Multiply
from tensorflow.keras.models import Model
from ssd_keras_layers.anchorBoxes import AnchorBoxes
from ssd_keras_layers.normalize import Normalize


def relu6(x):
    return ReLU(6)(x)

def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6) / 6.0


def conv_block(inputs, filters, kernel, strides, nl):
    channels_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, axis=channels_axis)(x)

    if nl == 'RE':
        x = Activation(relu6)(x)
    elif nl == "HS":
        x = Activation(hard_swish)(x)
    
    return x


def _squeeze(inputs):

    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x


def bottleneck(inputs, filters, kernel, e, strides, squeeze, nl, alpha=1):

    channels_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    t_channel = int(e)
    c_channel = int(alpha * filters)

    r = strides == (1, 1) and input_shape[3] == filters

    x = conv_block(inputs, t_channel, (1, 1), strides=(1, 1), nl=nl)

    x = DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, axis=channels_axis)(x)
    
    if nl == 'RE':
        x = Activation(relu6)(x)
    elif nl == "HS":
        x = Activation(hard_swish)(x)
    
    if squeeze:
        x = _squeeze(x)
    
    x = Conv2D(c_channel, (1, 1), padding='same')(x)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, axis=channels_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def ssd_conv(inputs, filters, kerner_size, padding='same', strides=(1, 1), name=None):

    x = Conv2D(filters, kerner_size, strides=strides, padding=padding, use_bias=False, name=name)(inputs)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, name=name+"/BN")(x)
    x = ReLU(6.)(x)
    return x


def SSD300(img_size, n_classes, l2_regularization=5e-4,
            anchors=[30, 60, 111, 162, 213, 264, 315],
            variances=[0.1, 0.1, 0.2, 0.2]):
    
    classes = n_classes + 1# Account for the background class.
    n_boxes = [4, 6, 6, 6, 4, 4]


    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = Input(shape=(img_size))
    alpha = 1.0

    conv1 = conv_block(x, 16, (3, 3), strides=(2, 2), nl='HS')

    block2 = bottleneck(conv1, 16, (3, 3), e=16, strides=(1, 1), squeeze=False, nl='RE')
    block3 = bottleneck(block2, 24, (3, 3), e=64, strides=(2, 2), squeeze=False, nl='RE')
    
    block4 = bottleneck(block3, 24, (3, 3), e=72, strides=(1, 1), squeeze=False, nl='RE')
    block5 = bottleneck(block4, 40, (5, 5), e=72, strides=(2, 2), squeeze=True, nl='RE')

    block6 = bottleneck(block5, 40, (5, 5), e=120, strides=(1, 1), squeeze=True, nl='RE')
    block7 = bottleneck(block6, 40, (5, 5), e=120, strides=(1, 1), squeeze=True, nl='RE')
    block8 = bottleneck(block7, 80, (3, 3), e=240, strides=(2, 2), squeeze=False, nl='HS')
    
    block9 = bottleneck(block8, 80, (3, 3), e=200, strides=(1, 1), squeeze=False, nl='HS')
    block10 = bottleneck(block9, 80, (3, 3), e=184, strides=(1, 1), squeeze=False, nl='HS')
    block11 = bottleneck(block10, 80, (3, 3), e=184, strides=(1, 1), squeeze=False, nl='HS')
    block12 = bottleneck(block11, 112, (3, 3), e=480, strides=(1, 1), squeeze=True, nl='HS')
    block13 = bottleneck(block12, 112, (3, 3), e=672, strides=(1, 1), squeeze=True, nl='HS')
    block14 = bottleneck(block13, 160, (5, 5), e=672, strides=(1, 1), squeeze=True, nl='HS')
    block15 = bottleneck(block14, 160, (5, 5), e=672, strides=(2, 2), squeeze=True, nl='HS')

    block16 = bottleneck(block15, 160, (5, 5), e=960, strides=(1, 1), squeeze=True, nl='HS')

    conv2 = conv_block(block16, 960, (1, 1), strides=(1, 1), nl='HS')
    conv3 = conv_block(conv2, 1280, (1, 1), strides=(1, 1), nl='HS')

    print("Conv3 shape:{}".format(conv3.shape))

    conv6_1 = ssd_conv(conv3, 256, (1, 1), padding='same', name='conv6_1')
    conv6_2 = ssd_conv(conv6_1, 512, (3, 3), strides=(2, 2), padding='same', name='conv6_2')
    
    conv7_1 = ssd_conv(conv6_2, 128, (1, 1), padding='same', name='conv7_1')
    conv7_2 = ssd_conv(conv7_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv7_2')
    
    conv8_1 = ssd_conv(conv7_2, 128, (1, 1), padding='same', name='conv8_1')
    conv8_2 = ssd_conv(conv8_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv8_2')
    
    conv9_1 = ssd_conv(conv8_2, 64, (1, 1), padding='same', name='conv9_1')
    conv9_2 = ssd_conv(conv9_1, 128, (3, 3), strides=(2, 2), padding='same', name='conv9_2')
    
    # Build SSD network

    conv4_3_norm = Normalize(20, name='conv4_3_norm')(block14)

    # Build "n_classes" confidence values for each box. Ouput shape: (b, h, w, n_boxes*n_classes)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * classes, (3, 3), padding='same',  name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * classes, (3, 3), padding='same', name='fc7_mbox_conf')(conv3)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * classes, (3, 3), padding='same',  name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)

    # Build 4 box coordinates for each box. Output shape: (b, h, w, n_boxes * 4)
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', name='fc7_mbox_loc')(conv3)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',  name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)
    
    # Generate the anchor boxes. Output shape: (b, h, w, n_boxes, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[0], max_size=anchors[1],aspect_ratios=[2],
                                             variances=variances,name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[1], max_size=anchors[2],aspect_ratios=[2,3],
                                             variances=variances,name='fc7_mbox_priorbox')(conv3)
    conv6_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[2], max_size=anchors[3],aspect_ratios=[2, 3],
                                             variances=variances, name='conv6_2_mbox_priorbox')(conv6_2)
    conv7_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[3], max_size=anchors[4],aspect_ratios=[2, 3],
                                             variances=variances, name='conv7_2_mbox_priorbox')(conv7_2)
    conv8_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[4], max_size=anchors[5],aspect_ratios=[2],
                                             variances=variances, name='conv8_2_mbox_priorbox')(conv8_2)
    conv9_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[5], max_size=anchors[6],aspect_ratios=[2],
                                        variances=variances, name='conv9_2_mbox_priorbox')(conv9_2)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Flatten(name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Flatten(name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Flatten(name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Flatten(name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Flatten(name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Flatten(name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Flatten(name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Flatten(name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Flatten(name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Flatten(name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Flatten(name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Flatten(name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    ### Concatenate the predictions from the different layers

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox,
                                                               conv7_2_mbox_priorbox,
                                                               conv8_2_mbox_priorbox,
                                                               conv9_2_mbox_priorbox])
    
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((-1, classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_loc,
                            mbox_conf,
                            mbox_priorbox])
    
    
    model = Model(inputs=x, outputs=predictions)
    return model
