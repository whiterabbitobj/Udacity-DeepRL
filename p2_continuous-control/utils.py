import os.path
import time
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
from agent import D4PG_Agent
from unityagents import UnityEnvironment
#from PIL import Image
from collections import deque
import torchvision.transforms as T



##########
## Saving & Loading
##########

def print_bracketing(statement):
    #mult = max(len(statement), 50)
    mult = 50
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}\n".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("\n{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    print("{}{}{}".format(upper, statement.center(mult), lower))

class Saver:
    def __init__(self, agent):
        # self.state_size = agent.state_size
        # self.action_size = agent.action_size
        self.file_ext = ".agent"
        self.save_dir = "saves"
        self._check_dir(self.save_dir)
        self.filename = self.generate_savename(agent.framework)
        print_bracketing("Saving to base filename: " + self.filename)
        #self.version = self.get_version()

    def _check_dir(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)

    def generate_savename(self, agent_name):
        """Generates an automatic savename for training files, will version-up as
           needed.
        """
        t = time.localtime()
        savename = "{}_{}_v".format(agent_name, time.strftime("%Y%m%d", time.localtime()))
        files = [f for f in os.listdir(self.save_dir)]# if os.path.isfile(self.save_dir+f)]# and os.path.splitext(f)[1] == self.file_ext]
        files = [f for f in files if savename in f]
        if len(files)>0:
            ver = [int(re.search("_v(\d+)", file).group(1)) for file in files]
            ver = max(ver) + 1
        else:
            ver = 1
        return "{}{}".format(savename, ver)

    def save_checkpoint(self, agent, save_every):
        """
        Saves the current Agent networks to checkpoint files.
        """
        if agent.episode % save_every:
            return
        checkpoint_dir = os.path.join(self.save_dir, self.filename)
        self._check_dir(checkpoint_dir)
        save_name = "{}_eps{}_ckpt{}".format(self.filename, agent.episode, self.file_ext)
        save_name = os.path.join(checkpoint_dir, save_name).replace('\\','/')
        statement = "Saving Agent checkpoint to: {}".format(save_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        torch.save(self._get_save_dict(agent), save_name)

    def save_final(self, agent):
        """
        Saves a checkpoint after training has finished.
        """
        save_name = "{}_eps{}_FINAL{}".format(self.filename, agent.episode-1, self.file_ext)
        save_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = "Saved final Agent weights to: {}".format(save_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        torch.save(self._get_save_dict(agent), save_name)

    def load_checkpoint(filepath, args):
        """
        Loads a checkpoint from an earlier trained agent.
        """
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        if checkpoint['agent_type'] == 'DQN':
            agent = Agent(checkpoint['state_size'], checkpoint['action_size'], args)
        if checkpoint['agent_type'] == 'D2DQN':
            agent = Agent(checkpoint['state_size'], checkpoint['action_size'], args)
        agent.q.load_state_dict(checkpoint['state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        args.num_episodes = 3
        return agent

    def _get_save_dict(self, agent):
        #agent.q.to('cpu')
        checkpoint = {'state_size': agent.state_size,
                      'action_size': agent.action_size,
                      'actor_dict': agent.actor.state_dict(),
                      'critic_dict': agent.critic.state_dict()
                      }
        return checkpoint

# def load_filepath(sep):
#     """Prompts the user about what save to load, or uses the last modified save.
#     """
#     files = [str(f) for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1] == '.pth']
#     if len(files) == 0:
#         print("Oops! Couldn't find any save files in the current directory.")
#         return None
#
#     files = sorted(files, key=lambda x: os.path.getmtime(x))
#     if args.latest:
#         print("{0}Proceeding with file: {1}\n{0}".format(sep, files[-1]))
#         return files[-1]
#     else:
#         message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
#         message = '\n'.join(message)
#         message = sep + message + " (LATEST)\n\nPlease choose a saved Agent training file (or: q/quit): "
#         save_file = input(message)
#         if save_file.lower() == "q" or save_file.lower() == "quit":
#             print("Quit before loading a file.")
#             return None
#         try:
#             file_index = len(files) - int(save_file)
#             if file_index < 0:
#                 raise Exception()
#             save_file = files[file_index]
#             print("{0}\nProceeding with file: {1}\n{0}".format(sep, save_file))
#             return save_file
#         except:
#             print("\nInput invalid...\n")
#             load_filepath()
