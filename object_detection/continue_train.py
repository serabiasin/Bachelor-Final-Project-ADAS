from __future__ import annotations
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import  ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from loss.ssd_loss import ssd_loss
import argparse

from generator import Generator
from utils.anchors_mobilenet import get_anchors_300
from utils.utils import BBoxUtility
from config import config


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenetv1', type=str,
                    help='Load architecture deep learning.')
parser.add_argument('--input', default='examples/*.png',
                    type=str, help='weight path for continue learning')
parser.add_argument('--lastepoch', default='1',
                    type=str, help='last Epoch')
args = parser.parse_args()

def learning_rate(epoch):
    if epoch < 50:
        return 1e-3
    elif epoch >= 50 and epoch < 100:
        return 1e-4
    elif epoch >= 100 and epoch < 150:
        return 1e-5
    else:
      return 1e-6

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists(config.MODEL_FOLDER):
    os.mkdir(config.MODEL_FOLDER)

if __name__ == "__main__":
    last_epoch = int(args.lastepoch)
    checkpoint_file = args.input
    if args.model == 'mobilenetv1':
        from models.ssd_mobilenetv1_300 import SSD300
    elif args.model == 'mobilenetv2':
        from models.ssd_mobilenetv2_300 import SSD300
    elif args.model == 'mobilenetv3':
        from models.ssd_mobilenetv3_300 import SSD300
    elif args.model == 'experiment':
        from models.ssd_experiment import SSD300        
        
    priors = get_anchors_300((config.IMAGE_SIZE_300[0], config.IMAGE_SIZE_300[1]),flag_features_map=args.model)
    
    bbox_util = BBoxUtility(len(config.CLASSES), priors)

    model = SSD300(config.IMAGE_SIZE_300,
                    n_classes=len(config.CLASSES),
                    anchors=config.ANCHORS_SIZE_300,
                    variances=config.VARIANCES)
    model.summary()
    checkpoint = ModelCheckpoint(config.MODEL_FOLDER + config.FILE_NAME+"-"+args.model + "-{epoch:02d}.h5",
                                 monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    reduce_lr = LearningRateScheduler(learning_rate, verbose=1)
    
    
    val_split = 0.1
    with open("train.txt") as f:
        lines = f.readlines()

    with open("val.txt") as f:
        val_lines = f.readlines()
    np.random.seed(1000)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines)


    gen = Generator(bbox_util, config.BATCH_SIZE, lines, val_lines, (config.IMAGE_SIZE_300[0], config.IMAGE_SIZE_300[1]), len(config.CLASSES))

    model.compile(optimizer=Adam(), loss=ssd_loss(len(config.CLASSES)).compute_loss)


    if checkpoint_file is not None:
        print("Continuing Training from", checkpoint_file)
        model.load_weights(checkpoint_file)
    
    model.fit_generator(gen.generator(True),
                        steps_per_epoch=len(lines)//config.BATCH_SIZE,
                        validation_data=gen.generator(True),
                        validation_steps=len(val_lines)//config.BATCH_SIZE,
                        epochs=config.EPOCHS,
                        initial_epoch=last_epoch,
                        callbacks=[checkpoint, reduce_lr])
    model.save(config.MODEL_FOLDER, save_format='tf')
