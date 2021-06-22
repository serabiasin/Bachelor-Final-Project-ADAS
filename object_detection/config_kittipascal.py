
class config:

    EPOCHS = 300
    BATCH_SIZE = 32
    IMAGE_SIZE_300 = (300, 300, 3)
    ANCHORS_SIZE_300 = [30, 60, 111, 162, 213, 264, 315] 
    VARIANCES = [0.1, 0.1, 0.2, 0.2]

    CLASSES = ['car', 'Van', 'Truck', 'person',
               'biker', 'Tram', 'motorbike', 'train', 'bus', 'bicycle']
    
    """
    JIKA DI TRAIN DI GOOGLE COLAB JANGAN LUPA, MODEL FOLDER NYA DI DIRECT 
    KE DRIVE!

    """
    TIPE = "KITTIPascal"
    DATASET = ["/kaggle/input/pascal-kitti-merged/"]
    # Dataset train, val, test images information.
    VOC_TEXT_FILE = ["/kaggle/input/pascal-kitti-merged/train.txt",
                     "/kaggle/input/pascal-kitti-merged/val.txt"]
    VOC_TRAIN_FILE = ["train.txt", "val.txt",] # Produce train text file.
    CONFIDENCE = 0.37
    NMS_IOU = 0.45
    # Stroage weigth folder
    MODEL_FOLDER = "/kaggle/working/object_detection/weights/"
    FILE_NAME = "SSD-"+TIPE  # Storage weigth weight
    

