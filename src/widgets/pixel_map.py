from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

import numpy as np
from skimage.transform import resize
from skimage import color


class Pixelmap(QLabel):
    #clicked = pyqtSignal(int, int, int) #pixel position y,x and mouse button
    moved = pyqtSignal(int,int) #pixel position y,x
    clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__()
        QLabel.__init__(self, parent)
        self.setMouseTracking(True) #track mouse movement even if no button is pressed
        self.pixmap = QPixmap() #pixelmap to display the image

        self.clickable = True
        self.default_zoom = 2.0
        self.zoom_factor = 1.0 #the current zoom factor
        self.inuse = False

        #self.geometry().height()
        self.image = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom),3), dtype=np.uint8) #HxWxC

        #link sliders
        self.slider_shiftx = None
        self.slider_shifty = None
        self.slider_zoom = None
        self.control_sliders = False

        self.updateCanvas()

    def empty(self):
        QLabel.clear(self)

        #reset
        self.zoom_factor = 1.0 #the current zoom factor
        self.inuse = False

        if self.slider_shiftx is not None:
            self.slider_shiftx.setEnabled(False)
        if self.slider_shifty is not None:
            self.slider_shifty.setEnabled(False)

        self.image = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom),3), dtype=np.uint8) #HxWxC
        self.updateCanvas()

    def setImage(self, image):
        self.image = image
        self.inuse = True

        fac = self.default_zoom

        shape = np.array(image.shape[:2])*fac
        leftover = (1024-shape)//fac

        #set offset slider range
        if self.slider_shiftx is not None:
            if self.control_sliders and leftover[1]>0:
                self.slider_shiftx.setRange(0,int(leftover[1]))
                self.slider_shiftx.setEnabled(True)
        if self.slider_shifty is not None:
            if self.control_sliders and leftover[0]>0:
                self.slider_shifty.setRange(0,int(leftover[0]))
                self.slider_shifty.setEnabled(True)

        self.updateCanvas()

    def updateCanvas(self):
        img = self.image.astype(np.uint8)
        val = 1.0
        if self.slider_zoom is not None:
            val = self.slider_zoom.value()
        fac = self.default_zoom*val

        if self.inuse:
            #ok select correct region
            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = self.slider_shiftx.value()

            shifty = 0
            if self.slider_shiftx is not None:
                shifty = self.slider_shiftx.value()

            img = img[shifty:shifty+1024//fac, shiftx:shiftx+1024//fac]

        #resize image
        img = resize(img, (img.shape[0]*fac, img.shape[1]*fac), order=0, preserve_range=True, anti_aliasing=False)

        #pad image if necessary!!!
        if img.shape[0]<1024:
            missing = 1024-img.shape[0]
            img = np.concatenate([img, np.zeros_like(img[:missing])], axis=0)

        if img.shape[1]<1024:
            missing = 1024-img.shape[1]
            img = np.concatenate([img, np.zeros_like(img[:,:missing])], axis=1)

        #draw new image onto label
        img = QImage(img.astype(np.uint8), img.shape[1], img.shape[0], 3*img.shape[1], QImage.Format.Format_RGB888)
        self.pixmap = self.pixmap.fromImage(img)
        QLabel.setPixmap(self, self.pixmap)

    def linkSliders(self, slider_shiftx=None, slider_shifty=None, slider_zoom=None, control_sliders=False):
        self.slider_shiftx = slider_shiftx
        self.slider_shifty = slider_shifty
        self.slider_zoom = slider_zoom
        self.control_sliders = control_sliders

        if self.slider_shiftx is not None:
            self.slider_shiftx.sliderReleased.connect(self.updateCanvas)
        if self.slider_shifty is not None:
            self.slider_shifty.sliderReleased.connect(self.updateCanvas)
        if self.slider_zoom is not None:
            self.slider_zoom.sliderReleased.connect(self.OnSliderZoom)

    def OnSliderZoom(self):
        val = 1.0
        if self.slider_zoom is not None:
            val = self.slider_zoom.value()
        fac = self.default_zoom*val

        shape = np.array(image.shape[:2])*fac
        leftover = (1024-shape)//fac

        #set offset slider range
        if self.slider_shiftx is not None:
            if self.control_sliders and leftover[1]>0:
                self.slider_shiftx.setRange(0,int(leftover[1]))
                self.slider_shiftx.setEnabled(True)
        if self.slider_shifty is not None:
            if self.control_sliders and leftover[0]>0:
                self.slider_shifty.setRange(0,int(leftover[0]))
                self.slider_shifty.setEnabled(True)

    """
    def mousePressEvent(self, ev: QMouseEvent):
        indy = int(ev.y()/4.0)
        indx = int(ev.x()/4.0)

        typ = ev.button() #left click=1, right click=2

        if self.clickable:
            self.clicked.emit(index)

    def mouseMoveEvent(self, ev: QMouseEvent):
        #convert positions to pixels first
        y = int(ev.y()/4.0)
        x = int(ev.x()/4.0)
        self.moved.emit(y, x) #convert to shifted coordinates
    """