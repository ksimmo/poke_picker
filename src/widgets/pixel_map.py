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
    clicked = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__()
        QLabel.__init__(self, parent)
        self.setMouseTracking(True) #track mouse movement even if no button is pressed
        self.pixmap = QPixmap() #pixelmap to display the image

        self.clickable = True
        self.default_zoom = 2.0
        self.zoom_factor = 1.0 #the current zoom factor
        self.inuse = False

        self.pokes = []
        self.poke_color = np.array([255,0,255], dtype=np.uint8)

        #self.geometry().height() -> currently we now that it is fixed to 1024
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

        self.pokes = []

        if self.slider_shiftx is not None:
            self.slider_shiftx.setEnabled(False)
        if self.slider_shifty is not None:
            self.slider_shifty.setEnabled(False)

        self.image = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom),3), dtype=np.uint8) #HxWxC
        self.updateCanvas()

    def setImage(self, image, pokes=None):
        self.image = image
        self.inuse = True

        if pokes is None:
            self.pokes = []
        else:
            self.pokes = pokes

        self.OnSliderZoom() #trigger change in zoom so shift sliders get recalculated

    def updateCanvas(self):
        img = self.image.astype(np.uint8)

        #draw annotations onto image
        for i in range(len(self.pokes)):
            img[self.pokes[i][1], self.pokes[i][0]] = self.poke_color

        val = 1.0
        if self.slider_zoom is not None:
            val = self.slider_zoom.value()
        fac = int(self.default_zoom*val)

        if self.inuse:
            #ok select correct region
            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = int(self.slider_shiftx.value())

            shifty = 0
            if self.slider_shifty is not None:
                shifty = int(self.slider_shifty.value())

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

    #connect sliders to widget (only 1 widget is allowed to also change sliders -> only set control_sliders true for 1 widget at the same time)
    #call this function in window constructor
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
        fac = int(self.default_zoom*val)

        shape = np.array(self.image.shape[:2])*fac
        leftover = shape-1024

        #we only allow shifting of original pixels
        leftover = leftover//fac

        #set offset slider range
        if self.slider_shiftx is not None:
            if self.control_sliders and leftover[1]>0:
                shiftx = self.slider_shiftx.value()
                self.slider_shiftx.blockSignals(True)
                self.slider_shiftx.setRange(0,int(leftover[1]))
                #TODO: adapt shift slider to new zoom
                self.slider_shiftx.setValue(0)
                self.slider_shiftx.blockSignals(False)
                self.slider_shiftx.setEnabled(True)
            elif self.control_sliders and leftover[1]<=0:
                self.slider_shiftx.setEnabled(False)
        if self.slider_shifty is not None:
            if self.control_sliders and leftover[0]>0:
                shifty = self.slider_shifty.value()
                self.slider_shifty.blockSignals(True)
                self.slider_shifty.setRange(0,int(leftover[0]))
                #TODO: adapt shift slider to new zoom
                self.slider_shifty.setValue(0) #sliders are not updated! why???
                self.slider_shifty.blockSignals(False)
                self.slider_shifty.setEnabled(True)
            elif self.control_sliders and leftover[0]<=0:
                self.slider_shifty.setEnabled(False)

        self.updateCanvas()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.inuse:
            #convert positions to pixels first
            val = 1.0
            if self.slider_zoom is not None:
                val = self.slider_zoom.value()
            fac = int(self.default_zoom*val)

            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = int(self.slider_shiftx.value())

            shifty = 0
            if self.slider_shifty is not None:
                shifty = int(self.slider_shifty.value())

            pos = ev.position()
            x = int(pos.x()/fac)+shiftx
            y = int(pos.y()/fac)+shifty
            self.moved.emit(x, y) #convert to shifted coordinates

    def mousePressEvent(self, ev: QMouseEvent):
        if self.inuse:
            #convert positions to pixels first
            val = 1.0
            if self.slider_zoom is not None:
                val = self.slider_zoom.value()
            fac = int(self.default_zoom*val)

            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = int(self.slider_shiftx.value())

            shifty = 0
            if self.slider_shifty is not None:
                shifty = int(self.slider_shifty.value())

            pos = ev.position()
            x = int(pos.x()/fac)+shiftx
            y = int(pos.y()/fac)+shifty

            action = ev.button() #left click=1, right click=2
            if action==Qt.MouseButton.LeftButton: #set annotation
                action = 1
            elif action==Qt.MouseButton.RightButton: #remove annotation
                action = 0
            else:
                action = 0

            #check if we are inside the image
            if self.clickable and x<self.image.shape[1] and y<self.image.shape[0]:
                self.clicked.emit(x, y, action)
                self.changePokes(x,y,action)

    def updatePokes(self, pokes=[]):
        self.pokes = pokes
        self.updateCanvas()

    def changePokes(self, x, y, action):
        found = False
        found_index = -1
        for i in range(len(self.pokes)):
            if self.pokes[i][0]==x and self.pokes[i][1]==y:
                found = True
                found_index = i
                break

        if not found and action==1:
            self.pokes.append(np.array([x,y], dtype=int))
        elif found and action==0:
            del self.pokes[found_index]

        self.updateCanvas()