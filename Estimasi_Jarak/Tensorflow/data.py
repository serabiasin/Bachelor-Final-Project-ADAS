import tensorflow as tf

from io import BytesIO
from zipfile import ZipFile
from sklearn.utils import shuffle
import os


class DataLoader():
    def __init__(self, csv_file='data/nyu2_train.csv', DEBUG=False):
        #height,width,channel
        self.shape_rgb = (480, 640, 3)
        self.shape_depth = (240, 320, 1)
        self.read_nyu_data(csv_file, DEBUG=DEBUG)

    def nyu_resize(self, img, resolution=480, padding=6):
        from skimage.transform import resize
        return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True)

    def read_nyu_data(self, csv_file, DEBUG=False):
        csv = open(csv_file, 'r').read()
        nyu2_train = list((row.split(',')
                           for row in (csv).split('\n') if len(row) > 0))

        # Dataset shuffling happens here
        nyu2_train = shuffle(nyu2_train, random_state=0)

        # Test on a smaller dataset
        if DEBUG:
            nyu2_train = nyu2_train[:10]

        # A vector of RGB filenames.
        self.filenames = []
        for i in nyu2_train:
          self.filenames.append(
              '/kaggle/input/kitti-depth-dataset'+i[0])

        # A vector of depth filenames.
        self.labels = []
        for i in nyu2_train:
          self.labels.append(
              '/kaggle/input/kitti-depth-dataset'+i[1])
        # Length of dataset
        self.length = len(self.filenames)

    def _parse_function(self, filename, label):
        # Read images from disk
        image_decoded = tf.image.resize(tf.image.decode_jpeg(
            tf.io.read_file(filename)), [self.shape_rgb[0], self.shape_rgb[1]])
        depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)), [
                                        self.shape_depth[0], self.shape_depth[1]])

        # Format
        rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        rgb = tf.clip_by_value(rgb/255, 0, 1)
        # depth = tf.image.convert_image_dtype(
        #     depth_resized / 255.0, dtype=tf.float32)

        # Normalize the depth values (in m)
        # depth = 80.0 / tf.clip_by_value(depth * 80.0, 1.0, 80.0)

        depth = 80.0 / tf.clip_by_value(depth_resized, 1.0, 80.0)

        return rgb, depth
        
    def get_batched_dataset(self, batch_size):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.filenames, self.labels))
        self.dataset = self.dataset.shuffle(buffer_size=len(
            self.filenames), reshuffle_each_iteration=True)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(
            map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size=batch_size)

        return self.dataset
