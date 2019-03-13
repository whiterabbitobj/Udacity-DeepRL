import os.path
import time
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment
from collections import deque
import torchvision.transforms as T

##########
## Interact with the environment
##########




class Environment():
    def __init__(self, args):
        self.device = args.device
        self.pixels = args.pixels
        self.training = args.train
        # self.phi = deque(maxlen=args.framestack*2)
        self.phi = deque(maxlen=args.framestack)

        if self.pixels:
            unity_filename = "VisualBanana_Windows_x86_64/Banana.exe"
        else:
            unity_filename = "Banana_Windows_x86_64/Banana.exe"
        self.env = UnityEnvironment(file_name=unity_filename, no_graphics=args.nographics)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]

        self.reset()
        self.nA = brain.vector_action_space_size

        self.state_size = list(self.state(reset=True).shape)
        print("STATE SIZE:", self.state_size)

    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return reward, done

    def reset(self):
        self.env_info = self.env.reset(train_mode=self.training)[self.brain_name]

    def close(self):
        self.env.close()

    def state(self, reset=False):
        if self.pixels:
            state = self.env_info.visual_observations[0]
            frame = self._process_frame(state)
            #frame = self._process_color_frame(state)
            self._stack(frame, reset)
            stack = self._get_stack().unsqueeze(0).to(self.device)
            return stack
        else:
            state = self.env_info.vector_observations[0]
            return torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    def _process_frame(self, state): #GRAYSCALE IMPLEMENTATION
        #frame = state.squeeze(0) * 255 #uncompress data from 0-1 to 0-255
        #frame = frame[35:-2,2:-2,:] #crop frame
        #frame = np.ascontiguousarray(frame, dtype=np.uint8) #ensure cropped data is not kept in memory
        frame = (state.squeeze(0) * 255).astype(np.uint8) #uncompress data from 0-1 to 0-255
        transforms = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
        frame = transforms(frame)
        return frame

    def _process_color_frame(self, state):
        frame = state.transpose(0,3,1,2)# * 255 #from NHWC to NCHW
        #frame = frame[:,:,35:-2,2:-2] # crop frame
        #frame = np.ascontiguousarray(frame, dtype=np.float32)# / 255 #ensure cropped data is not kept in memory
        frame = state.squeeze(0).astype(np.uint8)
        return torch.from_numpy(frame)

    def _stack(self, frame, reset):
        if reset:
            self.phi = deque([frame for i in range(self.phi.maxlen)], maxlen=self.phi.maxlen)
        else:
            self.phi.append(frame)
        return

    def _get_stack(self):
        # stack = torch.cat(list(self.phi)[::2],dim=0) #get the last frames over a timeperiod including skipped frames
        stack = torch.cat(list(self.phi),dim=0) #get the last frames over a timeperiod including skipped frames
        #stack = stack.transpose(1,0)
        return stack #.to(self.device)
