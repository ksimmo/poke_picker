import sys
import os
import time

from copy import deepcopy
import json

from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6 import uic

import numpy as np

import torch
import torch.nn.functional as F
#from torchcodec.decoders import VideoDecoder
#import decord
#decord.bridge.set_bridge('torch')
from torchvision.io import VideoReader
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

from src.utils.log import Logger

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode: str='sintel', padding_factor: int=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


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
        self.ui_label_flow.poke_color = np.array([0,0,0], dtype=np.uint8)

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

        self.ui_button_poke_add.clicked.connect(self.OnPokeButtonAdd)
        self.ui_button_poke_delete.clicked.connect(self.OnPokeButtonDelete)

        self.ui_button_calc.clicked.connect(self.OnVideoCalculate)
        self.ui_button_prev_t.clicked.connect(self.OnPrevFrame)
        self.ui_button_next_t.clicked.connect(self.OnNextFrame)
        self.ui_button_set_t.clicked.connect(self.OnSetFrame)
        self.ui_slider_t.sliderReleased.connect(self.OnSliderT)

        self.ui_label_video.moved.connect(self.OnMouseMove)
        self.ui_label_flow.moved.connect(self.OnMouseMove)
        self.ui_label_video.clicked.connect(self.OnMouseClicked)
        #self.ui_label_video.clicked.connect(self.ui_label_flow.changePokes)
        self.ui_label_flow.clicked.connect(self.OnMouseClicked)
        #self.ui_label_flow.clicked.connect(self.ui_label_video.changePokes)

        #stuff for handling video
        self.video_metadata = None
        self.video = None
        self.current_frame_index = 0
        self.flow = None
        self.pokes = []

        #RAFT for flow estimation
        self.raft_model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        self.raft_model.eval()
        for param in self.raft_model.parameters():
            param.requires_grad = False

    def log(self, text: str, status: int=0):
        if status==0:
            self.log_normal.write(text)
        elif status==1:
            self.log_err.write(text)

    #override
    def closeEvent(self, event):
        #close stuff here
        #####
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

                #add pokes
                for i in range(self.video_metadata["num_frames"]-1): #last frame does not have a flow to pick pokes from
                    self.pokes.append([]) #add an empty set pokes per frame

                self.ui_button_set_t.setEnabled(True)
                self.ui_button_next_t.setEnabled(True)
                self.ui_slider_t.setRange(0,self.video_metadata["num_frames"]-2) #last frame has no poke and indices start from 0
                #TODO: set slider t to 0
                self.ui_slider_t.setEnabled(True)
                self.ui_slider_zoom.setEnabled(True)
                self.SetCurrentFrame()

            except Exception as e:
                self.log("Cannot load video!", 1)
                print(e)

                self.video = None
                self.video_metadata = None
        else:
            self.log("Video {} does not exists!".format(path), 1)

    def OnVideoButtonUnload(self):
        self.video_metadata = None
        self.video = None
        self.current_frame_index = 0
        self.flow = None
        self.pokes = []

        #reset widgets

        self.ui_edit_t.blockSignals(True)
        self.ui_edit_t.setText("0")
        self.ui_edit_t.blockSignals(False)
        self.ui_button_set_t.setEnabled(False)
        self.ui_button_next_t.setEnabled(False)

        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(0)
        self.ui_slider_t.setRange(0,0)
        self.ui_slider_t.blockSignals(False)
        self.ui_slider_t.setEnabled(False)

        self.ui_slider_shiftx.blockSignals(True)
        self.ui_slider_shiftx.setValue(0)
        self.ui_slider_shiftx.setRange(0,0)
        self.ui_slider_shiftx.blockSignals(False)
        #disabling is done via pixelmap

        self.ui_slider_shifty.blockSignals(True)
        self.ui_slider_shifty.setValue(0)
        self.ui_slider_shifty.setRange(0,0)
        self.ui_slider_shifty.blockSignals(False)
        #disabling is done via pixelmap

        self.ui_slider_zoom.blockSignals(True)
        self.ui_slider_zoom.setValue(1)
        self.ui_slider_zoom.blockSignals(False)
        self.ui_slider_zoom.setEnabled(False)

        self.ui_label_video.empty()
        self.ui_label_flow.empty()

        self.ui_combo_poke.clear()
        self.ui_edit_poke1.setText("")
        self.ui_edit_poke2.setText("")

        self.ui_label_flowmag.setText("0")
        self.ui_label_pos.setText("[PosX,PosY]")

        self.log("Current video unloaded!")


    #poke loading/saving
    def OnPokeButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "JSON (*.json);;All files (*)")
        self.ui_edit_ppath.setText(fname[0])

    def OnPokeButtonSave(self):
        path = self.ui_edit_ppath.text()

        if not path.endswith(".json"):
            self.log("File for saving pokes needs to be a JSON (.json) file!", 1)
            return

        #write pokes to file
        try:
            f = open(path, "w")
            pokes = []
            for i in range(len(self.pokes)):
                pokes.append([])
                for j in range(len(self.pokes[i])):
                    pokes[i].append(self.pokes[i][j].tolist())
            json.dump(pokes, f)
            f.close()
            self.log("Successfully saved pokes to {}".format(path))
        except Exception as e:
            self.log("Failed saving pokes to {}".format(path), 1)
            print(e)

    def OnPokeButtonLoad(self):
        path = self.ui_edit_ppath.text()

        if not path.endswith(".json"):
            self.log("File for loading pokes needs to be a JSON (.json) file!", 1)
            return

        #load pokes from file
        try:
            f = open(path, "r")
            pokes = json.load(f)
            self.pokes = []
            for i in range(len(pokes)):
                self.pokes.append([])
                for j in range(len(pokes[i])):
                    self.pokes[i].append(np.array(pokes[i][j]))
                    if i==self.current_frame_index:
                        self.ui_combo_poke.addItem("[{},{},{},{}]".format(pokes[i][j][0],pokes[i][j][1],pokes[i][j][2],pokes[i][j][3]))
            f.close()
            self.log("Successfully loaded pokes from {}".format(path))
        except Exception as e:
            self.log("Failed loading pokes from {}".format(path), 1)
            print(e)
            
            #reset pokes
            for i in range(self.video_metadata["num_frames"]-1):
                self.pokes.append([]) #add an empty set pokes per frame
        
        if len(self.pokes)<self.video_metadata["num_frames"]-1:
            missing = self.video_metadata["num_frames"]-1-len(self.pokes)
            for i in range(missing):
                self.pokes.append([])
            self.log("Loaded pokes are too short! Appending empty poke sets.", 1)
        elif len(self.pokes)>self.video_metadata["num_frames"]-1:
            missing = len(self.pokes)-self.video_metadata["num_frames"]-1
            for i in reversed(range(missing)):
                del self.pokes[-1]
            self.log("Loaded pokes are too long! Removing additional poke sets.", 1)

        self.ui_label_video.updatePokes(self.pokes[self.current_frame_index])
        self.ui_label_flow.updatePokes(self.pokes[self.current_frame_index])

    def OnPokeButtonClear(self):
        self.pokes[self.current_frame_index] = []
        self.ui_label_video.updatePokes()
        self.ui_label_flow.updatePokes()
        self.ui_combo_poke.clear()
        self.log("Clearing current frame pokes ...")
    
    def OnPokeButtonClearAll(self):
        for i in range(len(self.pokes)):
            self.pokes[i] = []
        self.ui_label_video.updatePokes()
        self.ui_label_flow.updatePokes()
        self.ui_combo_poke.clear()
        self.log("Clearing all pokes ...")

    ###########################
    def OnVideoCalculate(self):
        if self.video is None:
            return
        #get device
        device = self.ui_combo_device.currentText()
        device = torch.device(device)

        rel_flow = self.ui_checkbox_rel_flow.isChecked()
        text = self.ui_edit_chunk_size.text()
        chunk_size = int(text)

        self.setEnabled(False)
        start_time = time.time()
        self.raft_model = self.raft_model.to(device)
        padder = InputPadder(self.video.size(), mode="sintel")

        flows = []
        if rel_flow:
            start_frame = self.video[0:1].to(device)
            start_frame = (start_frame.float()/255.0-0.5)/0.5 #map to [-1,1]
            start_frame = padder.pad(start_frame)[0]
        chunk_size = self.video.size(0) if chunk_size<=0 else chunk_size #we are measuring in flow frames!
        num_chunks = int(np.ceil(self.video.size(0)/chunk_size))
        for i in range(num_chunks):
            start = i*chunk_size
            end = min(self.video.size(0), (i+1)*chunk_size+1)
            if end-start==1: #for last frame no flow exists anyways
                break
            clip = self.video[start:end].to(device)
            clip = (clip.float()/255.0-0.5)/0.5 #map to [-1,1]
            clip = padder.pad(clip)[0]
            if not rel_flow:
                flow = self.raft_model(clip[0:-1], clip[1:])[-1]
            else:
                flow = self.raft_model(start_frame.repeat(clip.size(0)-1,1,1,1), clip)[-1]
            flow = padder.unpad(flow)
            flows.append(flow.cpu())
        self.flow = torch.cat(flows, dim=0)
        if rel_flow:
            self.flow = self.flow[1:] #remove first flow frame as it will be just 0
            del start_frame
        del clip

        end_time = time.time()
        self.setEnabled(True)

        self.log("Flow calculated (time taken: {:.3f}s)!".format(end_time-start_time))

        self.SetCurrentFrame()

    def OnPrevFrame(self):
        if self.current_frame_index-1>=0:
            next_frame = self.current_frame_index-1
        else:
            next_frame = self.video.size(0)-1 #start again from last frame
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(next_frame)
        self.ui_slider_t.blockSignals(False)

        self.current_frame_index = next_frame
        self.SetCurrentFrame() #update slider

    def OnNextFrame(self):
        if self.current_frame_index+1<self.video.size(0)-1:
            next_frame = self.current_frame_index+1
        else:
            next_frame = 0 #start again from first frame
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(next_frame)
        self.ui_slider_t.blockSignals(False)

        self.current_frame_index = next_frame
        self.SetCurrentFrame() #update slider

    def OnSetFrame(self):
        path = self.ui_edit_t.text()
        if not path.isnumeric():
            self.log("Frame index is not numeric!", 1)
            return
        
        index = int(path)
        if index<0 or index>=self.video.size(0)-1:
            self.log("Cannot set frame due to wrong length!",1)
            return
        
        if index!=self.current_frame_index:
            self.ui_slider_t.blockSignals(True)
            self.ui_slider_t.setValue(index)
            self.ui_slider_t.blockSignals(False)

            self.current_frame_index = index
            self.SetCurrentFrame() #update slider

    def OnSliderT(self):
        val = self.ui_slider_t.value()

        self.current_frame_index = val
        self.SetCurrentFrame()

    def SetCurrentFrame(self):
        if self.video is not None:
            #get current index
            frame = self.video[self.current_frame_index]
            self.ui_label_video.setImage(torch.movedim(frame, 0, 2).numpy(), self.pokes[self.current_frame_index])

            if self.flow is not None:
                flow = self.flow[self.current_frame_index]
                flow = flow_to_image(flow.unsqueeze(0).squeeze(0))
                self.ui_label_flow.setImage(torch.movedim(flow, 0, 2).numpy(), self.pokes[self.current_frame_index])
                mag = torch.sqrt(torch.sum(torch.pow(flow, 2),dim=1)).numpy()

            #add pokes to combobox
            self.ui_edit_poke1.setText("")
            self.ui_edit_poke2.setText("")
            self.ui_combo_poke.clear()
            for i in range(len(self.pokes[self.current_frame_index])):
                p = self.pokes[self.current_frame_index][i]
                self.ui_combo_poke.addItem("[{},{},{},{}]".format(p[0],p[1],p[2],p[3]))


    def OnMouseMove(self, x: int, y: int):
        self.ui_label_pos.setText("[{},{}]".format(x,y))
        if self.flow is not None:
            if x>=0 and x<self.flow.size(-1) and y>=0 and y<self.flow.size(-2):
                poke = self.flow[self.current_frame_index,:,y,x].numpy()
                pokemag = np.sqrt(np.sum(poke**2))
                self.ui_label_flowmag.setText("{}".format(int(pokemag)))

    def OnMouseClicked(self, x: int, y: int, action: int):
        found = False
        found_index = -1
        for i in range(len(self.pokes[self.current_frame_index])):
            if self.pokes[self.current_frame_index][i][0]==x and self.pokes[self.current_frame_index][i][1]==y:
                found = True
                found_index = i
                break

        if not found and action==1:
            self.log("Adding poke at x={} y={}".format(x, y))
            if self.flow is None:
                self.pokes[self.current_frame_index].append(np.array([x,y,0,0], dtype=int))
            else:
                flowx = int(self.flow[self.current_frame_index,0,y,x].item())
                flowy = int(self.flow[self.current_frame_index,1,y,x].item())
                self.pokes[self.current_frame_index].append(np.array([x,y,x+flowx,y+flowy], dtype=int))
            pokes = self.pokes[self.current_frame_index][-1]
            self.ui_combo_poke.addItem("[{},{},{},{}]".format(pokes[0], pokes[1], pokes[2], pokes[3]))
        elif found and action==0:
            del self.pokes[self.current_frame_index][found_index]
            self.ui_combo_poke.removeItem(found_index)
            self.log("Removing poke at x={} y={}".format(x, y))

        self.ui_label_video.updatePokes(self.pokes[self.current_frame_index])
        self.ui_label_flow.updatePokes(self.pokes[self.current_frame_index])

    def OnPokeButtonAdd(self):
        text1 = self.ui_edit_poke1.text()
        text2 = self.ui_edit_poke2.text()

        if not text1.isnumeric():
            self.log("Start position X is not numeric!", 1)
            return
        if not text2.isnumeric():
            self.log("Start position Y is not numeric!", 1)
            return
        
        pos1 = int(text1)
        pos2 = int(text2)

        if pos1<0 and pos1>self.video.size(-1):
            self.log("Start position X is out of bound!", 1)
            return
        if pos2<0 and pos2>self.video.size(-2):
            self.log("Start position Y is out of bound!", 1)
            return
        
        self.ui_label_video.pokes.append(np.array([pos1,pos2], dtype=int))
        self.ui_label_video.updateCanvas()
        self.ui_label_flow.pokes.append(np.array([pos1,pos2], dtype=int))
        self.ui_label_flow.updateCanvas()
        self.OnMouseClicked(pos1,pos2,1)   

    def OnPokeButtonDelete(self):
        if self.ui_combo_poke.count()>0:
            index = self.ui_combo_poke.currentIndex()
            text = self.ui_combo_poke.currentText()[1:-1] #remove []

            pos1 = text.find(",")
            x = int(text[:pos1])
            pos2 = text.find(",", pos1+1)
            y = int(text[pos1+1:pos2])
            
            self.OnMouseClicked(x,y,0) 

