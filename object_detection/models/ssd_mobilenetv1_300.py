
from tensorflow.keras.layers import Conv2D, Reshape, ZeroPadding2D, Concatenate, Input, Activation, Flatten, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.ops.gen_array_ops import pad
from ssd_keras_layers.anchorBoxes import AnchorBoxes
from ssd_keras_layers.normalize import Normalize


def depthwise_conv(inputs, pointwise_filter, alpha, depth_mutiplier=1, strides=(1, 1), name=None):

    pointwise_filter = int(pointwise_filter * alpha)
    
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_mutiplier, strides=strides,
                        use_bias=False, name=name+"/DW")(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=name+"/DW_BN")(x)
    x = ReLU(6., name=name+"/DW_RELU")(x)

    x = Conv2D(pointwise_filter, (1, 1), strides=(1, 1), padding='same', use_bias=False, name=name+"/PW")(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name=name+"/PW_BN")(x)
    x = ReLU(6., name=name+"PW_BN")(x)
    return x


def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    #x = ZeroPadding2D(((1, 1), (1, 1)), name="conv1/padding")(inputs)
    x = Conv2D(filters, kernel, strides=strides, padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001, name='conv1/BN')(x)
    x = ReLU(6., name='conv1/RELU')(x)
    return x

def ssd_conv(inputs, filters, kerner_size, padding='same', strides=(1, 1), l2_reg=5e-4, name=None):
    
    x = Conv2D(filters, kerner_size, strides=strides, use_bias=False, padding=padding, name=name)(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001)(x)
    x = ReLU(6.)(x)
    return x

def SSD300(img_size, n_classes, l2_reg=5e-4,
            anchors=[30, 60, 111, 162, 213, 264, 315],
            variances=[0.1, 0.1, 0.2, 0.2]):

    classes = n_classes + 1# Account for the background class.
    n_boxes = [4, 6, 6, 6, 4, 4]
    
    # Build network

    x = Input(shape=(img_size))

    conv = conv_block(x, 32, alpha=1, strides=(2, 2))
    block1 = depthwise_conv(conv, 64, alpha=1, name='block1')

    block2 = depthwise_conv(block1, 128, alpha=1, strides=(2, 2), name='block2')
    block3 = depthwise_conv(block2, 128, alpha=1, name='block3')

    block4 = depthwise_conv(block3, 256, alpha=1, strides=(2, 2), name='block4')
    block5 = depthwise_conv(block4, 256, alpha=1, name='block5')
    
    block6 = depthwise_conv(block5, 512, alpha=1, strides=(2, 2), name='block6')
    block7 = depthwise_conv(block6, 512, alpha=1, name='block7')
    block8 = depthwise_conv(block7, 512, alpha=1, name='block8')
    block9 = depthwise_conv(block8, 512, alpha=1, name='block9')
    block10 = depthwise_conv(block9, 512, alpha=1, name='block10')
    block11 = depthwise_conv(block10, 512, alpha=1, name='block11')

    block12 = depthwise_conv(block11, 1024, alpha=1, strides=(2, 2), name='block12')
    block13 = depthwise_conv(block12, 1024, alpha=1, name='block13')

    print("Shape:{}".format(block13.shape))
    
    conv6_1 = ssd_conv(block13, 256, (1, 1), padding='same', name='conv6_1')
    conv6_2 = ssd_conv(conv6_1, 512, (3, 3), strides=(2, 2), padding='same', name='conv6_2')
    
    conv7_1 = ssd_conv(conv6_2, 128, (1, 1), padding='same', name='conv7_1')
    conv7_2 = ssd_conv(conv7_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv7_2')
    
    conv8_1 = ssd_conv(conv7_2, 128, (1, 1), padding='same', name='conv8_1')
    conv8_2 = ssd_conv(conv8_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv8_2')
    
    conv9_1 = ssd_conv(conv8_2, 64, (1, 1), padding='same', name='conv9_1')
    conv9_2 = ssd_conv(conv9_1, 128, (3, 3), strides=(2, 2), padding='same', name='conv9_2')

    # Build SSD network

    conv4_3_norm = Normalize(20, name='conv4_3_norm')(block11)

    # Build "n_classes" confidence values for each box. Ouput shape: (b, h, w, n_boxes*n_classes)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * classes, (3, 3), padding='same',  name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * classes, (3, 3), padding='same', name='fc7_mbox_conf')(block13)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * classes, (3, 3), padding='same',  name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)

    # Build 4 box coordinates for each box. Output shape: (b, h, w, n_boxes * 4)
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', name='fc7_mbox_loc')(block13)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',  name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)
    
    # Generate the anchor boxes. Output shape: (b, h, w, n_boxes, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[0], max_size=anchors[1],aspect_ratios=[2],
                                             variances=variances,name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[1], max_size=anchors[2],aspect_ratios=[2,3],
                                             variances=variances,name='fc7_mbox_priorbox')(block13)
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
