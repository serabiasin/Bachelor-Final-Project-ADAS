import xml.etree.ElementTree as ET
from config import config
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='default',
                    type=str, help='Dataset type to config')
args = parser.parse_args()

if args.dataset=='default':
    print("default loaded")
    from config import config
elif args.dataset=='kitti_pascal':
    print('Kitti Pascal loaded')
    from config_kittipascal import config
elif args.dataset=='pascal':
    print('Pascal loaded')
    from config_voc import config

def convert_annotation(xml_file, text_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find('name').text
        if cls not in config.CLASSES or int(difficult) == 1:
            continue
        cls_id = config.CLASSES.index(cls)
        bndbox = obj.find('bndbox')
        bbox = (int(float(bndbox.find("xmin").text)), int(float(bndbox.find("ymin").text)), int(float(bndbox.find("xmax").text)), int(float(bndbox.find("ymax").text)))
        text_file.write(" " + ",".join([str(b) for b in bbox]) + "," + str(cls_id))

if __name__ == "__main__":
    
    for index, file_name in enumerate(config.VOC_TRAIN_FILE):
        print(config.VOC_TEXT_FILE[index])
        with open(config.VOC_TEXT_FILE[index]) as f:
            lines = f.read().split()
        np.random.shuffle(lines)
        f = open(file_name, 'w')
        
        for line in lines:
            print(line)
            f.write(config.DATASET[0] + "JPEGImages/" + line[:-4] + ".jpg")
            convert_annotation(config.DATASET[0] + "Annotations/" + line[:-4] + ".xml", f)
            f.write("\n")
        f.close()
