
from tensorflow.keras.layers import Conv2D, Reshape, ZeroPadding2D, Concatenate, Input, Activation, Flatten, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.ops.gen_array_ops import pad
from ssd_keras_layers.anchorBoxes import AnchorBoxes
from ssd_keras_layers.normalize import Normalize
import tensorflow as tf


def ssd_convolution(inputs, filters, kerner_size, padding='same', strides=(1, 1), l2_reg=5e-4, name=None):

    x = Conv2D(filters, kerner_size, strides=strides,
               use_bias=False, padding=padding, name=name)(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=0.00001)(x)
    x = ReLU(6.)(x)
    return x


def SSD300(img_size, n_classes, l2_reg=5e-4,
           anchors=[30, 60, 111, 162, 213, 264, 315],
           variances=[0.1, 0.1, 0.2, 0.2]):

    classes = n_classes + 1  # Account for the background class.
    n_boxes = [4, 6, 6, 6, 4, 4]

    # Build network

    temp = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size), include_top=False, weights=None)
    base_network = Model(
        inputs=temp.input, outputs=temp.get_layer("top_conv").output)

    conv6_1= ssd_convolution(base_network.get_layer("block7a_project_conv").output, 256, (1, 1), padding='same', name='conv6_1')
    conv6_2= ssd_convolution(conv6_1,  512, (3, 3), strides=(2, 2), padding='same', name='conv6_2')

    conv7_1=ssd_convolution(conv6_2, 128, (1, 1), padding='same', name='conv7_1')
    conv7_2=ssd_convolution(conv7_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv7_2')

    conv8_1=ssd_convolution(conv7_2, 128, (1, 1), padding='same', name='conv8_1')
    conv8_2=ssd_convolution(conv8_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv8_2')

    conv9_1=ssd_convolution(conv8_2, 128, (1, 1), padding='same', name='conv9_1')
    conv9_2=ssd_convolution(conv9_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv9_2')


    conv4_3_norm = Normalize(classes-1, name='conv4_3_norm')(base_network.get_layer('block6d_project_conv').output)

    # Build "n_classes" confidence values for each box. Ouput shape: (b, h, w, n_boxes*n_classes)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * classes, (3, 3), padding='same',  name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * classes, (3, 3), padding='same', name='fc7_mbox_conf')(base_network.get_layer("block7a_project_conv").output)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * classes, (3, 3), padding='same',  name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)

    # Build 4 box coordinates for each box. Output shape: (b, h, w, n_boxes * 4)
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', name='fc7_mbox_loc')(base_network.get_layer("block7a_project_conv").output)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',  name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)

    # Generate the anchor boxes. Output shape: (b, h, w, n_boxes, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[0], max_size=anchors[1],aspect_ratios=[2],
                                                variances=variances,name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[1], max_size=anchors[2],aspect_ratios=[2,3],
                                                variances=variances,name='fc7_mbox_priorbox')(base_network.get_layer("block7a_project_conv").output)
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
                            
    model = Model(inputs=base_network.input, outputs=predictions)

    return model
