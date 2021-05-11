
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Reshape, ZeroPadding2D, Concatenate, Input, Activation, Flatten, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, add
from tensorflow.keras.models import Model
from ssd_keras_layers.anchorBoxes import AnchorBoxes
from ssd_keras_layers.normalize import Normalize

def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(inputs, kernel_size):

    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0]//2, kernel_size[1]//2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
    

def _inverted_res_block(inputs, expansion, strides, filters, alpha=1.0, block_id=None):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, (1, 1), padding='same', use_bias=False, name=prefix + "expand")(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "expand_BN")(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    
    # Depthwise
    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3), name=prefix + 'pad')(x)
    
    x = DepthwiseConv2D((3, 3), strides=strides, use_bias=False, 
                        padding='same' if strides == 1 else 'valid', 
                        name=prefix + "depthwise")(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "depthwise_BN")(x)
    x = ReLU(6., name=prefix + "depthwise_relu")(x)

    x = Conv2D(pointwise_filters, (1, 1), padding='same', use_bias=False, name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_conv_filters and strides == 1:
        return add([inputs, x])
    return x


def ssd_conv(inputs, filters, kerner_size, padding='same', strides=(1, 1), l2_reg=5e-4, name=None):
    
    x = Conv2D(filters, kerner_size, strides=strides, use_bias=False, padding=padding, name=name)(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001)(x)
    x = ReLU(6.)(x)
    return x


def SSD300(img_size, n_classes, 
            anchors=[30, 60, 111, 162, 213, 264, 315],
            variances=[0.1, 0.1, 0.2, 0.2]):

    classes = n_classes + 1# Account for the background class.
    n_boxes = [4, 6, 6, 6, 4, 4]
    
    # Build network

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = Input(shape=(img_size))
    alpha = 1.0

    first_block_filters = _make_divisible(32 * alpha, 8)
    stage1 = ZeroPadding2D(padding=correct_pad(x, 3), name='conv1_pad')(x)
    stage1 = Conv2D(first_block_filters, 3, strides=(2, 2), padding='same', use_bias=False, name='conv1')(x)
    stage1 = BatchNormalization(axis=bn_axis, name="conv1_bn")(stage1)
    stage1 = ReLU(6., name="conv1_relu")(stage1)
    stage1 = _inverted_res_block(stage1, filters=16, alpha=alpha, strides=1, expansion=1, block_id=0)
    
    stage2 = _inverted_res_block(stage1, filters=24, alpha=alpha, strides=2, expansion=6, block_id=1)
    stage2 = _inverted_res_block(stage2, filters=24, alpha=alpha, strides=1, expansion=6, block_id=2)

    stage3 = _inverted_res_block(stage2, filters=32, alpha=alpha, strides=2, expansion=6, block_id=3)
    stage3 = _inverted_res_block(stage3, filters=32, alpha=alpha, strides=1, expansion=6, block_id=4)
    stage3 = _inverted_res_block(stage3, filters=32, alpha=alpha, strides=1, expansion=6, block_id=5)
    
    stage4 = _inverted_res_block(stage3, filters=64, alpha=alpha, strides=2, expansion=6, block_id=6)
    stage4 = _inverted_res_block(stage4, filters=64, alpha=alpha, strides=1, expansion=6, block_id=7)
    stage4 = _inverted_res_block(stage4, filters=64, alpha=alpha, strides=1, expansion=6, block_id=8)
    stage4 = _inverted_res_block(stage4, filters=64, alpha=alpha, strides=1, expansion=6, block_id=9)
    stage4 = _inverted_res_block(stage4, filters=96, alpha=alpha, strides=1, expansion=6, block_id=10)
    stage4 = _inverted_res_block(stage4, filters=96, alpha=alpha, strides=1, expansion=6, block_id=11)
    stage4 = _inverted_res_block(stage4, filters=96, alpha=alpha, strides=1, expansion=6, block_id=12)

    stage5 = _inverted_res_block(stage4, filters=160, alpha=alpha, strides=2, expansion=6, block_id=13)
    stage5 = _inverted_res_block(stage5, filters=160, alpha=alpha, strides=1, expansion=6, block_id=14)
    stage5 = _inverted_res_block(stage5, filters=160, alpha=alpha, strides=1, expansion=6, block_id=15)
    stage5 = _inverted_res_block(stage5, filters=320, alpha=alpha, strides=1, expansion=6, block_id=16)

    stage5 = Conv2D(1280, (1, 1), use_bias=False, name='conv_last')(stage5)
    stage5 = BatchNormalization(axis=bn_axis, epsilon=1e-3, momentum=0.999, name='conv_last_bn')(stage5)
    stage5 = ReLU(6., name='out_relu')(stage5)

    print("Stage5 shape:{}".format(stage5.shape))

    conv6_1 = ssd_conv(stage5, 256, (1, 1), padding='same', name='conv6_1')
    conv6_2 = ssd_conv(conv6_1, 512, (3, 3), strides=(2, 2), padding='same', name='conv6_2')
    
    conv7_1 = ssd_conv(conv6_2, 128, (1, 1), padding='same', name='conv7_1')
    conv7_2 = ssd_conv(conv7_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv7_2')
    
    conv8_1 = ssd_conv(conv7_2, 128, (1, 1), padding='same', name='conv8_1')
    conv8_2 = ssd_conv(conv8_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv8_2')
    
    conv9_1 = ssd_conv(conv8_2, 64, (1, 1), padding='same', name='conv9_1')
    conv9_2 = ssd_conv(conv9_1, 128, (3, 3), strides=(2, 2), padding='same', name='conv9_2')
    
    # Build SSD network

    conv4_3_norm = Normalize(20, name='conv4_3_norm')(stage4)

    # Build "n_classes" confidence values for each box. Ouput shape: (b, h, w, n_boxes*n_classes)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * classes, (3, 3), padding='same',  name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * classes, (3, 3), padding='same', name='fc7_mbox_conf')(stage5)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * classes, (3, 3), padding='same',  name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)

    # Build 4 box coordinates for each box. Output shape: (b, h, w, n_boxes * 4)
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', name='fc7_mbox_loc')(stage5)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',  name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)
    
    # Generate the anchor boxes. Output shape: (b, h, w, n_boxes, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[0], max_size=anchors[1],aspect_ratios=[2],
                                             variances=variances,name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[1], max_size=anchors[2],aspect_ratios=[2,3],
                                             variances=variances,name='fc7_mbox_priorbox')(stage5)
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
