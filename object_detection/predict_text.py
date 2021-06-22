from ssd_predict import detector
from PIL import Image
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='experiment', type=str,
                    help='Load architecture deep learning.')
parser.add_argument('--input', default='examples/*.png',
                    type=str, help='weight path for continue learning')
args = parser.parse_args()
###########################
NETWORK = args.model # Choose an option in `mobilenetv1`, `mobilenetv2` or `mobilenetv3`
WEIGHT_FILE = args.input # The network weigth file path.
###########################

if not os.path.exists("results"):
    os.mkdir("results")

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"

det = detector(weight_path=WEIGHT_FILE, network=NETWORK)

predicts = []
with open("val.txt") as f:
    line = f.readline()
    while line:
        l = line.split()
        predicts.append(l[0])
        line = f.readline()
result_file = open(os.path.join("results","{}_result.txt".format(NETWORK)), "w")

for index, pre in enumerate(predicts):
    img = pre
    print(img)

    image = Image.open(img)
    results = det.detected_bbox(image)

    result_file.write(img)
    for result in results:
        result_file.write(" {},{},{},{},{}".format(result[0], result[1], result[2], result[3], result[4]))
    
    result_file.write("\n")
    
result_file.close()
print(os.path.join("results", "{}_result.txt".format(NETWORK)))
