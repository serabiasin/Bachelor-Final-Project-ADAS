# Keras / TensorFlow
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from utils import predict, load_images, display_images
from layers import BilinearUpSampling2D
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D,
                  'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model('/content/DenseDepth/kitti.h5',
                   custom_objects=custom_objects, compile=False)

# Input images
inputs = np.asarray(Image.open(
    "/content/DenseDepth/examples/sampel1.png"), dtype=float)
print(inputs.shape)
inputs = inputs.reshape(1, 480, 640, 3)
print(inputs.shape)
# Compute results (hasil ini sebenernya sudah bisa digunakan (satuan cm))
outputs = model.predict(inputs)

rescaled = outputs[0][:, :, 0]
rescaled = rescaled - np.min(rescaled)
rescaled = rescaled / np.max(rescaled)

#image depth map
gambar_plasma = plasma(rescaled)[:, :, :3]
