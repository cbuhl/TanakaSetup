import sys, time, os
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt

import cv2

def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):
    '''
    With infinite thanks to verified.human at
    https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image
    '''
    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)

class simple(QMainWindow):
    def __init__(self):
        super(simple, self).__init__()
        loadUi('mainWindow.ui', self)
        self.image = None
        self.capture = 0 #To check if we have started acq.

        #Connect signals for camera interaction
        self.btnCameraStart.clicked.connect(self.start_cam)
        self.btnCameraStop.clicked.connect(self.stop_cam)
        self.btnCameraUpdate.clicked.connect(self.update_cam)

        #define signals for events to ensure the sectioning does not crash
        self.nImgN.valueChanged.connect(self.protect_sectioning)
        self.nImgX.valueChanged.connect(self.protect_sectioning)
        self.nImgY.valueChanged.connect(self.protect_sectioning)

        self.frame_width = 1280
        self.frame_height = 720

    def print_log(self):
        print('test')


    def start_cam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)

        #Define a fast timer for the live images.
        self.timerFast = QTimer(self)
        self.timerFast.timeout.connect(self.update_frame)
        self.timerFast.start(self.tExposure.value()+50)

        #Define a slow timer for the histograms.
        self.timerSlow = QTimer(self)
        self.timerSlow.timeout.connect(self.update_histograms)
        self.timerSlow.start(300)


    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        dispy = self.nImgY.value()
        dispx = self.nImgX.value()
        n = self.nImgN.value()
        #Define the sectioning
        xa1 = self.frame_width//4 - n//2
        xa2 = self.frame_width//4 + n//2
        ya1 = self.frame_height//2 - n//2
        ya2 = self.frame_height//2 + n//2

        dispx = self.nImgX.value()
        xb1 = self.frame_width//4*3 - n//2 + dispx
        xb2 = self.frame_width//4*3 + n//2 + dispx
        yb1 = self.frame_height//2 - n//2 + dispy
        yb2 = self.frame_height//2 + n//2 + dispy

        #Do the sectioning
        self.imgA = self.image[ya1:ya2, xa1:xa2].copy()
        self.imgB = self.image[yb1:yb2, xb1:xb2].copy()
        self.section_diff = ((self.imgA/2 - self.imgB/2) + 128).astype(np.uint8)
        self.section_color = apply_custom_colormap(self.section_diff, 'bwr')

        #Add the rectangles to show the data nPosition
        cv2.rectangle(self.image,(xa1,ya1),(xa2,ya2),(255,0,0),3)
        cv2.rectangle(self.image,(xb1,yb1),(xb2,yb2),(0,0,255),3)
        self.display_raw(self.image)
        self.display_diff(self.section_color)


    def display_raw(self, img):
        qformat = QImage.Format_Indexed8
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)

        self.rawLabel.setPixmap(QPixmap.fromImage(outImage))
        self.rawLabel.setScaledContents(True)


    def display_diff(self, img):
        qformat = QImage.Format_RGB888
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)

        self.diffLabel.setPixmap(QPixmap.fromImage(outImage))
        self.diffLabel.setScaledContents(True)


    def update_histograms(self):

        histA = np.histogram(self.imgA, bins=255, range=(0,255))
        histB = np.histogram(self.imgB, bins=255, range=(0,255))

        img = np.ones((257,256,3), dtype=np.uint8)*255

        #Draw a red line for the A image
        histx = histA[1][1:]
        histy = histA[0]
        ptsA = np.array([histx, histy/histy.max()*250], dtype = np.int32).T
        ptsA = ptsA.reshape((-1,1,2))
        cv2.polylines(img, [ptsA], False, color=(255,0,0))

        #Draw a red line for the B image
        histx = histB[1][1:]
        histy = histB[0]
        ptsA = np.array([histx, histy/histy.max()*250], dtype = np.int32).T
        ptsA = ptsA.reshape((-1,1,2))
        cv2.polylines(img, [ptsA], False, color=(0,0,255))

        self.display_histogram(cv2.flip(img,0))


    def display_histogram(self, img):
        qformat = QImage.Format_RGB888
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)

        self.histLabel.setPixmap(QPixmap.fromImage(outImage))
        self.histLabel.setScaledContents(True)


    def stop_cam(self):
        if type(self.capture) != int: #Check to make sure the acquisition has started.
            self.timerFast.stop()
            self.timerSlow.stop()
            self.capture.release()


    def update_cam(self):
        self.stop_cam()
        self.start_cam()


    def protect_sectioning(self):
        #the aim is to avoid the following values to reach a larger size than
        # the self.frame_width and self.frame_height
        n = self.nImgN.value()

        self.nImgX.setMaximum(self.frame_width//4 - n//2 - 1)
        self.nImgX.setMinimum(n//2 - self.frame_width//4*3 +1)
        self.nImgY.setMaximum(self.frame_height//2 - n//2 - 1)
        self.nImgY.setMinimum(n//2 - self.frame_height//2 +1)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window=simple()
    window.setWindowTitle('V1')
    window.show()


    sys.exit(app.exec_())
