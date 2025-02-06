import sys
import os
import time

from copy import deepcopy

from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6 import uic

import torch
#from torchcodec.decoders import VideoDecoder
#import decord
#decord.bridge.set_bridge('torch')
from torchvision.io import VideoReader
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

from src.utils.log import Logger
#from src.worker.main_thread import MainThread

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("src/gui/mainwindow.ui", self)

        self.log_normal = Logger(self.ui_textedit_console, sys.stdout)
        self.log_err = Logger(self.ui_textedit_console, sys.stderr, QColor(255, 0, 0))

        #self.main_thread = MainThread(None)
        #self.main_thread.start()

        #query devices
        num_gpus = torch.cuda.device_count()
        self.ui_combo_device.addItem("cpu")
        for i in range(num_gpus):
            self.ui_combo_device.addItem("cuda:{}".format(i))

        #setup widgets
        self.ui_label_video.linkSliders(self.ui_slider_shiftx, self.ui_slider_shifty, self.ui_slider_zoom, control_sliders=True)
        self.ui_label_flow.linkSliders(self.ui_slider_shiftx, self.ui_slider_shifty, self.ui_slider_zoom)
        self.ui_slider_t.setRange(0,0)
        self.ui_slider_shiftx.setEnabled(False)
        self.ui_slider_shifty.setEnabled(False)
        self.ui_slider_zoom.setEnabled(False)

        #register callbacks
        self.ui_button_vselect.clicked.connect(self.OnVideoButtonSelect)
        self.ui_button_vload.clicked.connect(self.OnVideoButtonLoad)
        self.ui_button_vunload.clicked.connect(self.OnVideoButtonUnload)

        self.ui_button_pselect.clicked.connect(self.OnPokeButtonSelect)
        self.ui_button_psave.clicked.connect(self.OnPokeButtonSave)
        self.ui_button_pload.clicked.connect(self.OnPokeButtonLoad)
        self.ui_button_pclear.clicked.connect(self.OnPokeButtonClear)
        self.ui_button_pclear_all.clicked.connect(self.OnPokeButtonClearAll)

        self.ui_edit_range_start.editingFinished.connect(self.OnRangeEditStart)
        self.ui_edit_range_end.editingFinished.connect(self.OnRangeEditEnd)
        self.ui_button_set_range.clicked.connect(self.OnButtonSetRange)
        self.ui_button_calc.clicked.connect(self.OnVideoCalculate)
        self.ui_button_next_t.clicked.connect(self.OnNextFrame)
        self.ui_button_set_t.clicked.connect(self.OnSetFrame)

        #stuff for handling video
        self.video_metadata = None
        self.video = None
        self.video_start = 0
        self.video_end = -1
        self.current_frame_index = 0
        self.flow = None

        #RAFT for flow estimation
        self.raft_model = raft_large(pretrained=True)
        self.raft_model.eval()
        for param in self.raft_model.parameters():
            param.requires_grad = False

    def log(self, text, status=0):
        if status==0:
            self.log_normal.write(text)
        elif status==1:
            self.log_err.write(text)

    #override
    def closeEvent(self, event):
        #stop threads

        # close other stuff
        QtWidgets.QMainWindow.closeEvent(self, event)

    #video loading
    def OnVideoButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "MP4 (*.mp4);;GIF (*.gif);;All files (*)")
        self.ui_edit_vpath.setText(fname[0])

    def OnVideoButtonLoad(self):
        self.OnVideoButtonUnload() #make sure everything is unloaded before
        path = self.ui_edit_vpath.text()

        if os.path.exists(path):
            try:
                self.log("Loading video from {} ...".format(path))
                #vr = decord.VideoReader(path)
                #num_frames = len(vr)
                #fps = vr.get_avg_fps()
                vr = VideoReader(path, "video")
                self.video_metadata = deepcopy(vr.get_metadata()["video"])

                frames = []
                for frame in vr:
                    frames.append(frame['data'])
                self.video = torch.stack(frames)
                self.video_metadata["num_frames"] = self.video.size(0)
                self.video_metadata["fps"] = self.video_metadata["fps"][0] 
                print(self.video_metadata)
                #vd = VideoDecoder(path)
                #self.video_metadata = deepcopy(vd.metadata)
                self.log("Video info: num_frames={} | fps={:.1f} H={} W={}".format(self.video_metadata["num_frames"], self.video_metadata["fps"], 
                                                                                   self.video.size(-2), self.video.size(-1)))
                del vr

                #adjust range
                self.ui_edit_range_end.setText("{}".format(self.video_metadata["num_frames"]))
                self.video_end = self.video_metadata["num_frames"]
                self.ui_button_set_t.setEnabled(True)
                self.ui_button_next_t.setEnabled(True)
                self.ui_slider_t.setRange(0,self.video_metadata["num_frames"]-1)
                self.ui_slider_t.setEnabled(True)
                self.ui_slider_zoom.setEnabled(True)
                self.SetCurrentFrame()

            except Exception as e:
                self.log("Cannot load video!", 1)
                self.log(str(e), 1)

                self.video = None
                self.video_metadata = None
        else:
            self.log("Video {} does not exists!".format(path), 1)

    def OnVideoButtonUnload(self):
        self.video_metadata = None
        self.video = None
        self.video_start = 0
        self.video_end = -1
        self.current_frame_index = 0
        self.flow = None

        #reset widgets
        self.ui_edit_range_start.setText("0")
        self.ui_edit_range_end.setText("-1")

        self.ui_slider_t.setEnabled(False)
        self.ui_button_set_t.setEnabled(False)
        self.ui_button_next_t.setEnabled(False)
        self.ui_slider_t.setRange(0,0)

        self.ui_slider_shiftx.blockSignals(True)
        self.ui_slider_shiftx.setValue(0)
        self.ui_slider_shiftx.blockSignals(False)

        self.ui_slider_shifty.blockSignals(True)
        self.ui_slider_shifty.setValue(0)
        self.ui_slider_shifty.blockSignals(False)

        self.ui_slider_zoom.blockSignals(True)
        self.ui_slider_zoom.setValue(1)
        self.ui_slider_zoom.blockSignals(False)
        self.ui_slider_zoom.setEnabled(False)

        self.ui_label_video.empty()
        self.ui_label_flow.empty()

        self.log("Current video unloaded!")


    #poke loading/saving
    def OnPokeButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "JSON (*.json);;All files (*)")
        self.ui_edit_ppath.setText(fname[0])

    def OnPokeButtonSave(self):
        path = self.ui_edit_ppath.text()

    def OnPokeButtonLoad(self):
        path = self.ui_edit_ppath.text()

    def OnPokeButtonClear(self):
        pass
    
    def OnPokeButtonClearAll(self):
        pass

    ###########################
    def OnRangeEditStart(self):
        if self.video is None:
            self.ui_edit_range_start.setText("0")
        else:
            text = self.ui_edit_range_start.text()
            number = int(text)
            if number>self.video.size(0)-1:
                self.log_err("Video only has {} frames!".format(self.video_metadata["num_frames"]))
                self.ui_edit_range_start.setText("0")

    def OnRangeEditEnd(self):
        if self.video is None:
            self.ui_edit_range_end.setText("0")
        else:
            text = self.ui_edit_range_end.text()
            number = int(text)
            if number>self.video.size(0):
                self.log_err("Video only has {} frames!".format(self.video_metadata["num_frames"]))
                self.ui_edit_range_end.setText("0")

    def OnButtonSetRange(self):
        if self.video is not None:
            start = self.ui_edit_range_start.text()
            end = self.ui_edit_range_end.text()
            start = int(start)
            end = int(end)
            self.video_start = start
            self.video_end = end if end==-1 else self.video.size(0)

            self.log("Setting range to [{}:{}]! This will influence already set pokes!".format(start, end))

            self.ui_slider_t.setEnabled(False)
            self.ui_slider_t.setRange(0,self.video_end-self.video_start-1)
            self.ui_slider_t.setEnabled(True)

    def OnVideoCalculate(self):
        if self.video is None:
            return
        #get device
        device = self.ui_combo_device.currentText()
        device = torch.device(device)

        rel_flow = self.ui_checkbox_rel_flow.isChecked()

        self.setEnabled(False)
        start = time.time()
        self.raft_model = self.raft_model.to(device)
        clip = self.video[self.video_start:self.video_end].to(device)
        clip = (clip.float()/255.0-0.5)/0.5 #map to [-1,1]
        flow = self.raft_model(clip[0:-1], clip[1:])[-1] if rel_flow else self.raft_model(clip[0:1].repeat(clip.size(0)-1,1,1,1), clip[1:])[-1]
        self.flow = flow.cpu()
        del clip
        end = time.time()
        self.setEnabled(True)

        self.log("Flow calculated (time taken: {:.3f}s)!".format(end-start))

        self.SetCurrentFrame()

    def OnNextFrame(self):
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(index)
        self.ui_slider_t.blockSignals(False)

    def OnSetFrame(self):
        path = self.ui_edit_t.text()
        index = int(path)
        if index<0 or index>=(self.video_end-self.video_start):
            self.log_err("Cannot set frame due to wrong length!")
            return

        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(index)
        self.ui_slider_t.blockSignals(False)

    def SetCurrentFrame(self):
        if self.video is not None:
            #get current index
            frame = self.video[self.current_frame_index]
            self.ui_label_video.setImage(torch.movedim(frame, 0, 2).numpy())

            if self.flow is not None:
                flow = self.flow[self.current_frame_index]
                flow = flow_to_image(flow.unsqueeze(0).squeeze(0))
                self.ui_label_flow.setImage(torch.movedim(flow, 0, 2).numpy())
