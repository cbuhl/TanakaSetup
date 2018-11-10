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

        self.btnCameraStart.clicked.connect(self.start_cam)
        self.btnCameraStop.clicked.connect(self.stop_cam)

    def start_cam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        pass

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        x1 = int(1280/4)
        x2 = int(1280/4*3)
        y1 = int(720/2)
        self.imgA = self.image[y1-128:y1+128, x1-128:x1+128].copy()
        self.imgB = self.image[y1-128:y1+128, x2-128:x2+128].copy()
        self.section_diff = ((self.imgA/2 - self.imgB/2) + 128).astype(np.uint8)
        self.section_color = apply_custom_colormap(self.section_diff, 'bwr')
        self.display_raw(self.image, 1)
        self.display_diff(self.section_color, 1)

    def display_raw(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3: #[0] rows, [1] columns, [2] channels
            if img.shape[2] == 4: #r,g,b,a
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #Need to convert from BGR to RGB.
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.rawLabel.setPixmap(QPixmap.fromImage(outImage))
            self.rawLabel.setScaledContents(True)

    def display_diff(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3: #[0] rows, [1] columns, [2] channels
            if img.shape[2] == 4: #r,g,b,a
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #Need to convert from BGR to RGB.
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.diffLabel.setPixmap(QPixmap.fromImage(outImage))
            self.diffLabel.setScaledContents(True)

    def stop_cam(self):
        self.timer.stop()
        pass




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window=simple()
    window.setWindowTitle('Simple Webcam Streamer')
    window.show()


    sys.exit(app.exec_())
