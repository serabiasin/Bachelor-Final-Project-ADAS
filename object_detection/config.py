
class config:

    EPOCHS = 400
    BATCH_SIZE = 16
    IMAGE_SIZE_300 = (300, 300, 3)
    ANCHORS_SIZE_300 = [50, 80, 131, 162, 213, 264, 315]  # Kitti SSD300
    VARIANCES = [0.1, 0.1, 0.2, 0.2]

    CLASSES = [ 'car', 'Van', 'Truck','person', 'biker', 'Tram']
    
    DATASET = ["/kaggle/input/kitti-pascal-kaggle/"]
    # Dataset train, val, test images information.
    VOC_TEXT_FILE = ["./kaggle/input/kitti-pascal-kaggle/ImageSets/Main/training.txt",
                     "/kaggle/input/kitti-pascal-kaggle/ImageSets/Main/validation.txt"]
    VOC_TRAIN_FILE = ["./train.txt", "./val.txt",] # Produce train text file.
    CONFIDENCE = 0.40
    NMS_IOU = 0.45
    # Stroage weigth folder
    MODEL_FOLDER = "/kaggle/working/SSD-Mobilenet/weights"
    FILE_NAME = "SSD-KITTI-Experiment" # Storage weigth weight
    

