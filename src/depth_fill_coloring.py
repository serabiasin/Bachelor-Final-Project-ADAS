# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#

import scipy
from skimage.color import rgb2gray
import numpy as np
from scipy.sparse.linalg import spsolve
from PIL import Image
import cv2
import os
from pathlib import Path
import cv2


class KittiDepthFill:
    def __init__(self,Rawpath=None,FolderDepthAnnotated=None,OutputFolder=None,checkpointfill=None):

        self.Rawpath=Rawpath
        self.FolderDepthAnnotated=FolderDepthAnnotated
        self.OutputFolder=OutputFolder
        self.listTrainFile=[]
        self.listValFile=[]
        self.checkpointfill = checkpointfill
    
    def getlistTrain(self):
        return self.listTrainFile

    def getlistVal(self):
        return self.listValFile
        
    def beginFill(self,filenameDepth):
        RGBPath = Path(filenameDepth)
        #extract tree folder
        folder_rgb=RGBPath.parts[6]
        filename_rgb=RGBPath.parts[10]
        rootfolder_rgb = RGBPath.parts[6][:10]

        fullpath_raw=os.path.join(self.Rawpath,rootfolder_rgb)
        fullpath_raw=os.path.join(fullpath_raw,folder_rgb)
        fullpath_raw = os.path.join(fullpath_raw, "image_03/data")
        fullpath_raw = os.path.join(fullpath_raw,filename_rgb)
        if os.path.isfile(fullpath_raw):
            filled_img = self.fill_depth_colorization(
            imgRgb=fullpath_raw, imgDepthInput=filenameDepth)
            cv2.imwrite(os.path.join(self.OutputFolder, filename_rgb),filled_img)
        print(fullpath_raw)

    def scanFolder(self):
        #scan folder annotated
        train_annotatedDepthPath=[]
        val_annotatedDepthPath=[]        
        folderanotasi=["train","val"]
        counter=0
        for path_train in folderanotasi:
            scanFilepath = os.path.join(self.FolderDepthAnnotated, path_train)
            for folder_stage2 in os.listdir(scanFilepath):
                if counter==0:
                    train_annotatedDepthPath.append(os.path.join(scanFilepath, folder_stage2))
                elif counter==1:
                    val_annotatedDepthPath.append(os.path.join(scanFilepath, folder_stage2))
            counter+=1

        train_annotatedDepthPath.sort()
        val_annotatedDepthPath.sort()
        
        #the hardest part, summoning .png files each folder
        extra_path_annotated="proj_depth/groundtruth/image_03"

        #train dataset
        for path in train_annotatedDepthPath:
            for lidar_annotated in os.listdir(os.path.join(path,extra_path_annotated)):
                buffer_path = os.path.join(path, extra_path_annotated)
                #check file or directory
                if os.path.isfile(os.path.join(buffer_path,lidar_annotated)):
                   filenameDepth=os.path.join(buffer_path, lidar_annotated)
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
    """

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


    def fill_depth_colorization(self,imgRgb=None, imgDepthInput=None, alpha=1):
        depthImg = cv2.imread(imgDepthInput)
        depthImg = cv2.cvtColor(depthImg, cv2.COLOR_BGR2GRAY)
        rgbImg = cv2.imread(imgRgb)

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
                c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

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

        new_vals = spsolve(A, b)
        new_vals = np.reshape(new_vals, (H, W), 'F')

        #print ('Done.')

        denoisedDepthImg = new_vals * maxImgAbsDepth

        output = denoisedDepthImg.reshape((H, W)).astype('float32')

        output = np.multiply(output, (1-knownValMask)) + depthImg

        return output
