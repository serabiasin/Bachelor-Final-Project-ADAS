import scipy
from skimage.color import rgb2gray
import numpy as np
from scipy.sparse.linalg import spsolve
from PIL import Image
from sparse_dot_mkl import sparse_qr_solve_mkl
import cv2
import os
from pathlib import Path
import cv2
import pandas as pd


class KittiDepthFill:
    def __init__(self, Rawpath=None, FolderDepthAnnotated=None, OutputFolder=None, ProcessedRawpath=None):

        self.Rawpath = Rawpath
        self.FolderDepthAnnotated = FolderDepthAnnotated
        self.OutputFolder = OutputFolder
        self.listTrainFile = []
        self.listValFile = []
        self.checkpointfill = OutputFolder
        self.RGBProcessedImage = ProcessedRawpath

        self.RGBProcessedImageTrain = []
        self.RGBProcessedImageVal = []

        self.bufferRGBProcessedImageTrain = []
        self.bufferRGBProcessedImageVal = []

        # print("Scan file...")
        # self.scanFolder()

    def getlistTrain(self):
        return self.listTrainFile

    def getlistVal(self):
        return self.listValFile

    def getlistRGBVal(self):
        return self.RGBProcessedImageVal

    def getlistRGBTrain(self):
        return self.RGBProcessedImageTrain

    def sanityCheckTrainRGB(self):
        counter = 0
        for path_file in os.listdir(os.path.join(self.RGBProcessedImage, "train")):
            if os.path.isfile(os.path.join(os.path.join(os.path.join(self.RGBProcessedImage, "train"), path_file))):
                counter = counter+1
                self.bufferRGBProcessedImageTrain.append(os.path.join(
                    os.path.join(os.path.join(self.RGBProcessedImage, "train"), path_file)))
        print("Jumlah gambar train %d / %d" %
              (counter+1, len(self.RGBProcessedImageTrain)))
        self.bufferRGBProcessedImageTrain.sort()
        self.bufferRGBProcessedImageTrain = np.array(
            self.bufferRGBProcessedImageTrain)
        # fullpath=os.path.join(self.checkpointfill,"listProcessedRGb.txt")
        # np.savetxt(fullpath, self.bufferRGBProcessedImageTrain, delimiter=',')

    def sanityCheckValRGB(self):
        counter = 0
        for path_file in os.listdir(os.path.join(self.RGBProcessedImage, "val")):
            if os.path.isfile(os.path.join(os.path.join(os.path.join(self.RGBProcessedImage, "val"), path_file))):
                counter = counter+1
                self.bufferRGBProcessedImageVal.append(os.path.join(
                    os.path.join(os.path.join(self.RGBProcessedImage, "val"), path_file)))
        print("Jumlah gambar val %d / %d" %
              (counter+1, len(self.RGBProcessedImageVal)))
        self.bufferRGBProcessedImageVal.sort()
        self.bufferRGBProcessedImageVal = np.array(
            self.bufferRGBProcessedImageVal)

    def sanityCheckTrain(self):
        counter = 0
        for path_file in os.listdir(os.path.join(self.OutputFolder, "train")):
            if os.path.isfile(os.path.join(os.path.join(os.path.join(self.OutputFolder, "train"), path_file))):
                counter = counter+1
        print("Jumlah gambar train %d / %d" %
              (counter+1, len(self.listTrainFile)))

    def beginFill(self, filenameDepth, flag):

        RGBPath = Path(filenameDepth)

        #extract tree folder
        folder_rgb = RGBPath.parts[7]
        filename_rgb = RGBPath.parts[11]
        rootfolder_rgb = RGBPath.parts[7][:10]

        fullpath_raw = os.path.join(self.Rawpath, rootfolder_rgb)
        fullpath_raw = os.path.join(fullpath_raw, folder_rgb)
        fullpath_raw = os.path.join(fullpath_raw, "image_03/data")
        fullpath_raw = os.path.join(fullpath_raw, filename_rgb)
        print(fullpath_raw)
        if os.path.isfile(fullpath_raw):

            filled_img = self.fill_depth_colorization(
                imgRgb=fullpath_raw, imgDepthInput=filenameDepth)

            #train
            if flag == 0:
                output_train = os.path.join(self.OutputFolder, "train")

                namabaru_file = folder_rgb+"###"+filename_rgb
                temp = os.path.join(output_train, namabaru_file)

                cv2.imwrite(temp, filled_img)
            elif flag == 1:
                output_val = os.path.join(self.OutputFolder, "val")

                namabaru_file = folder_rgb+"###"+filename_rgb
                temp = os.path.join(output_val, namabaru_file)

                cv2.imwrite(temp, filled_img)
        return temp

    #untuk hasil depth yang sudah diproses
    def scanFilled(self):
        #scan folder annotated
        filledTrain = []
        filledVal = []
        folderanotasi = ["train", "val"]
        counter = 0

        for path_train in folderanotasi:
            scanFilepath = os.path.join(self.OutputFolder, path_train)
            for folder_stage2 in os.listdir(scanFilepath):
                if counter == 0:
                    print(os.path.join(scanFilepath, folder_stage2))
                    self.RGBProcessedImageTrain.append(
                        os.path.join(scanFilepath, folder_stage2))
                elif counter == 1:
                    self.RGBProcessedImageVal.append(
                        os.path.join(scanFilepath, folder_stage2))
            counter += 1
        self.RGBProcessedImageVal.sort()
        self.RGBProcessedImageTrain.sort()

        self.RGBProcessedImageVal = np.array(self.RGBProcessedImageVal)
        self.RGBProcessedImageTrain = np.array(self.RGBProcessedImageTrain)

    def scanFolder(self):
        #scan folder annotated
        train_annotatedDepthPath = []
        val_annotatedDepthPath = []
        folderanotasi = ["train", "val"]
        counter = 0
        for path_train in folderanotasi:
            scanFilepath = os.path.join(self.FolderDepthAnnotated, path_train)
            for folder_stage2 in os.listdir(scanFilepath):
                if counter == 0:
                    train_annotatedDepthPath.append(
                        os.path.join(scanFilepath, folder_stage2))
                elif counter == 1:
                    val_annotatedDepthPath.append(
                        os.path.join(scanFilepath, folder_stage2))
            counter += 1

        train_annotatedDepthPath.sort()
        val_annotatedDepthPath.sort()

        #the hardest part, summoning .png files each folder
        extra_path_annotated = "proj_depth/groundtruth/image_03"

        #train dataset
        for path in train_annotatedDepthPath:
            for lidar_annotated in os.listdir(os.path.join(path, extra_path_annotated)):
                buffer_path = os.path.join(path, extra_path_annotated)
                #check file or directory
                if os.path.isfile(os.path.join(buffer_path, lidar_annotated)):
                   filenameDepth = os.path.join(buffer_path, lidar_annotated)
                   self.listTrainFile.append(filenameDepth)

        #val dataset
        for path in val_annotatedDepthPath:
            for lidar_annotated in os.listdir(os.path.join(path, extra_path_annotated)):
                buffer_path = os.path.join(path, extra_path_annotated)
                #check file or directory
                if os.path.isfile(os.path.join(buffer_path, lidar_annotated)):
                   filenameDepth = os.path.join(buffer_path, lidar_annotated)
                   self.listValFile.append(filenameDepth)

        self.listTrainFile.sort()
        self.listValFile.sort()

        self.listTrainFile = np.array(self.listTrainFile)
        self.listValFile = np.array(self.listValFile)

    def fillTrain(self):
        iteration = 1
        checkpoint_file = os.path.join(
            self.checkpointfill, "checkpoint_train.txt")
        if os.path.exists(checkpoint_file):
            f = open(checkpoint_file, 'r')
            path_temp = f.read()
            print(path_temp[:])
            f.close()

            index_data = np.where(self.listTrainFile == str(path_temp))[0][0]
            iteration = index_data+1

            for path_file in self.listTrainFile[index_data:]:
                out_folder = self.beginFill(path_file, 0)
                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistTrain()), out_folder))

                iteration = iteration+1
        else:
            for path_file in self.getlistTrain():
                out_folder = self.beginFill(path_file, 0)

                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistTrain()), out_folder))

                iteration = iteration+1

    def fillVal(self):
        iteration = 1
        checkpoint_file = os.path.join(
            self.checkpointfill, "checkpoint_val.txt")
        if os.path.exists(checkpoint_file):
            f = open(checkpoint_file, 'r')
            path_temp = f.read()
            print(path_temp[:])
            f.close()

            index_data = np.where(self.listValFile == str(path_temp))[0][0]
            iteration = index_data+1

            for path_file in self.listValFile[index_data:]:
                out_folder = self.beginFill(path_file, 1)
                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistVal()), out_folder))

                iteration = iteration+1
        else:
            for path_file in self.listValFile:
                out_folder = self.beginFill(path_file, 1)

                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistVal()), out_folder))

                iteration = iteration+1

    def copyAndResizeImageTrain(self):
        iteration = 1
        checkpoint_file = os.path.join(
            self.checkpointfill, "resizedphoto_train.txt")
        if os.path.exists(checkpoint_file):
            f = open(checkpoint_file, 'r')
            path_temp = f.read()
            print(path_temp[:])
            f.close()

            index_data = np.where(
                self.RGBProcessedImageTrain == str(path_temp))[0][0]
            iteration = index_data+1

            for path_file in self.RGBProcessedImageTrain[index_data:]:
                out_folder = self.beginresize(path_file, 0)
                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistRGBTrain()), out_folder))

                iteration = iteration+1
        else:
            for path_file in self.RGBProcessedImageTrain:
                out_folder = self.beginresize(path_file, 0)

                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistRGBTrain()), out_folder))

                iteration = iteration+1

    def copyAndResizeImageVal(self):
        iteration = 1
        checkpoint_file = os.path.join(
            self.checkpointfill, "resizedphoto_val.txt")
        if os.path.exists(checkpoint_file):
            f = open(checkpoint_file, 'r')
            path_temp = f.read()
            print(path_temp[:])
            f.close()

            index_data = np.where(
                self.RGBProcessedImageVal == str(path_temp))[0][0]
            iteration = index_data+1

            for path_file in self.RGBProcessedImageVal[index_data:]:
                out_folder = self.beginresize(path_file, 1)
                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistRGBVal()), out_folder))

                iteration = iteration+1
        else:
            for path_file in self.RGBProcessedImageVal:
                out_folder = self.beginresize(path_file, 1)

                with open(checkpoint_file, "w") as text_file:
                    text_file.write(path_file)
                text_file.close()

                print("data ke  %d / %d : %s" %
                      (iteration, len(self.getlistRGBVal()), out_folder))

                iteration = iteration+1

    def beginresize(self, filenameDepth, flag, scaling=35):

        RGBPath = Path(filenameDepth)

        #extract tree folder
        buffer = RGBPath.parts[-1]
        buffer2 = buffer.split('###')
        folder_rgb = buffer2[0]
        filename_rgb = buffer2[1]
        rootfolder_rgb = buffer2[0][:10]

        fullpath_raw = os.path.join(self.Rawpath, rootfolder_rgb)
        fullpath_raw = os.path.join(fullpath_raw, folder_rgb)
        fullpath_raw = os.path.join(fullpath_raw, "image_03/data")
        fullpath_raw = os.path.join(fullpath_raw, filename_rgb)
        print(fullpath_raw)
        if os.path.isfile(fullpath_raw):
            rgbImg = cv2.imread(fullpath_raw)
            width = 744
            height = 224
            dim = (width, height)
            print(dim)
            rgbImg = cv2.resize(rgbImg, dim, interpolation=cv2.INTER_NEAREST)
            #train
            if flag == 0:
                output_train = os.path.join(self.RGBProcessedImage, "train")

                namabaru_file = folder_rgb+"__"+filename_rgb
                temp = os.path.join(output_train, namabaru_file)

                cv2.imwrite(temp, rgbImg)
            elif flag == 1:
                output_val = os.path.join(self.RGBProcessedImage, "val")

                namabaru_file = folder_rgb+"__"+filename_rgb
                temp = os.path.join(output_val, namabaru_file)

                cv2.imwrite(temp, rgbImg)
        return temp

    def convertToCSV(self):
        self.sanityCheckTrainRGB()
        self.sanityCheckValRGB()
        trainsumbu_y=self.RGBProcessedImageTrain
        valsumbu_y=self.RGBProcessedImageVal

        trainsumbu_x = self.bufferRGBProcessedImageTrain
        valsumbu_x=self.bufferRGBProcessedImageVal
        
        df = pd.DataFrame({"X_train": trainsumbu_x, "Y_train": trainsumbu_y})
        fullpath_training=os.path.join(self.OutputFolder,'training_path.csv')
        df.to_csv(fullpath_training, index=False)
        print("data saved :",fullpath_training)

        df1 = pd.DataFrame({"X_val": valsumbu_x, "Y_val": valsumbu_y})
        fullpath_validation=os.path.join(self.OutputFolder,'validation_path.csv')
        df1.to_csv(fullpath_validation, index=False)
        print("validation saved:",fullpath_validation)


    """
    Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html


    Python port of depth filling code from NYU toolbox
    Speed needs to be improved

    Preprocesses the kinect depth image using a gray scale version of the
    RGB image as a weighting for the smoothing. This code is a slight
    adaptation of Anat Levin's colorization code:

    See: www.cs.huji.ac.il/~yweiss/Colorization/

    Args:
    imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
        be between 0 and 1.
    imgDepth - HxW matrix, the depth image for the current frame in
        absolute (meters) space.
    alpha - a penalty value between 0 and 1 for the current depth values.

    """

    def fill_depth_colorization(self, imgRgb=None, imgDepthInput=None, alpha=1, scaling=70):
        depthImg = cv2.imread(imgDepthInput)
        depthImg = cv2.cvtColor(depthImg, cv2.COLOR_BGR2GRAY)
        rgbImg = cv2.imread(imgRgb)

        width = int(rgbImg.shape[1] * (100-scaling) / 100)
        height = int(rgbImg.shape[0] * (100-scaling) / 100)
        dim = (width, height)

        rgbImg = cv2.resize(rgbImg, dim)
        depthImg = cv2.resize(depthImg, dim, interpolation=cv2.INTER_NEAREST)

        imgIsNoise = (depthImg == 0)
        maxImgAbsDepth = np.max(depthImg)
        imgDepth = depthImg / maxImgAbsDepth
        imgDepth[imgDepth > 1] = 1
        (H, W) = imgDepth.shape
        numPix = H * W
        indsM = np.arange(numPix).reshape((W, H)).transpose()
        knownValMask = (imgIsNoise == False).astype(int)
        grayImg = rgb2gray(rgbImg)
        winRad = 1
        len_ = 0
        absImgNdx = 0
        len_window = (2 * winRad + 1) ** 2
        len_zeros = numPix * len_window

        cols = np.zeros(len_zeros) - 1
        rows = np.zeros(len_zeros) - 1
        vals = np.zeros(len_zeros) - 1
        gvals = np.zeros(len_window) - 1

        for j in range(W):
            for i in range(H):
                nWin = 0
                for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                    for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                        if ii == i and jj == j:
                            continue

                        rows[len_] = absImgNdx
                        cols[len_] = indsM[ii, jj]
                        gvals[nWin] = grayImg[ii, jj]

                        len_ = len_ + 1
                        nWin = nWin + 1

                curVal = grayImg[i, j]
                gvals[nWin] = curVal
                c_var = np.mean(
                    (gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

                csig = c_var * 0.6
                mgv = np.min((gvals[:nWin] - curVal) ** 2)
                if csig < -mgv / np.log(0.01):
                    csig = -mgv / np.log(0.01)

                if csig < 2e-06:
                    csig = 2e-06

                gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
                gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
                vals[len_ - nWin:len_] = -gvals[:nWin]

                # Now the self-reference (along the diagonal).
                rows[len_] = absImgNdx
                cols[len_] = absImgNdx
                vals[len_] = 1  # sum(gvals(1:nWin))

                len_ = len_ + 1
                absImgNdx = absImgNdx + 1

        vals = vals[:len_]
        cols = cols[:len_]
        rows = rows[:len_]
        A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

        rows = np.arange(0, numPix)
        cols = np.arange(0, numPix)
        vals = (knownValMask * alpha).transpose().reshape(numPix)
        G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

        A = A + G
        b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

        #print ('Solving system..')

        new_vals = sparse_qr_solve_mkl(A, b)
        new_vals = np.reshape(new_vals, (H, W), 'F')

        #print ('Done.')

        denoisedDepthImg = new_vals * maxImgAbsDepth

        output = denoisedDepthImg.reshape((H, W)).astype('float32')

        output = np.multiply(output, (1-knownValMask)) + depthImg

        return output
