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
    mult = 50
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}\n".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("\n{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    if type(statement) is not list:
        statement = [statement]
    print(upper)
    for line in statement:
        print(line.center(mult))
    print(lower)

class Saver:
    def __init__(self, agent, args):
        # self.state_size = agent.state_size
        # self.action_size = agent.action_size
        self.file_ext = ".agent"
        self.save_dir = args.save_dir
        self.cwd = os.getcwd()
        self.latest = args.latest
        self.user_quit_message = "User quit process before loading a file."

        if not args.eval:
            self.filename = self.generate_savename(agent.framework)
            self._check_dir(self.save_dir)
            print_bracketing("Saving to base filename: " + self.filename)

        os.chdir(os.path.join(self.cwd, self.save_dir))

    def _check_dir(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)

    def generate_savename(self, agent_name):
        """Generates an automatic savename for training files, will version-up as
           needed.
        """
        t = time.localtime()
        savename = "{}_{}_v".format(agent_name, time.strftime("%Y%m%d", time.localtime()))
        files = [f for f in os.listdir()]# if os.path.isfile(self.save_dir+f)]# and os.path.splitext(f)[1] == self.file_ext]
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
        full_name = os.path.join(checkpoint_dir, save_name).replace('\\','/')
        statement = "Saving Agent checkpoint to: {}".format(full_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        torch.save(self._get_save_dict(agent), save_name)

    def save_final(self, agent):
        """
        Saves a checkpoint after training has finished.
        """
        save_name = "{}_eps{}_FINAL{}".format(self.filename, agent.episode-1, self.file_ext)
        full_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = "Saved final Agent weights to: {}".format(full_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        torch.save(self._get_save_dict(agent), save_name)

    def load_agent(self, agent):
        files = self._get_files()
        assert len(files) > 0, "Could not find any files to load in requested directory: {}".format(self.save_dir)
        if self.latest:
            file = self.load_checkpoint(agent, files[-1])
        else:
            filepath = self._get_filepath(files)
            file = self.load_checkpoint(agent, filepath)
        statement = "Successfully loaded file: {}".format(file)
        print_bracketing(statement)

    def load_checkpoint(self, agent, file):
        """
        Loads a checkpoint from an earlier trained agent.
        """
        # filepath = os.path.join(self.save_dir, file)
        checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
        agent.actor.load_state_dict(checkpoint['actor_dict'])
        agent.critic.load_state_dict(checkpoint['critic_dict'])
        agent._hard_update(agent.actor, agent.actor_target)
        agent._hard_update(agent.critic, agent.critic_target)
        return os.path.join(self.save_dir, file)

    def quit(self):
        os.chdir(self.cwd)

    def _get_files(self):
        # files = [str(f) for f in os.listdir() if os.path.isfile(f)]
        file_list = []
        for root, _, files in os.walk('.'):
            for file in files:
                file_list.append(os.path.join(root, file)[2:])
        #files = [os.path.join(root, file) for root, _, files in os.walk('.')]
        return sorted(file_list, key=lambda x: os.path.getmtime(x))

    def _get_filepath(self, files):
        """
        Prompts the user about what save to load, or uses the last modified save.
        """
        prompt = " (LATEST)\n\nPlease choose a saved Agent training file (or: q/quit): "
        message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
        message = '\n'.join(message)
        message = message + prompt
        save_file = input(message)
        if save_file.lower() in ("q", "quit"):
            # print("Quit before loading a file.")
            # return None
            raise KeyboardInterrupt(self.user_quit_message)
        try:
            file_index = len(files) - int(save_file)
            assert file_index >= 0
            # if file_index < 0:
            #     raise Exception()
            return files[file_index]
        except:
            print("\nInput invalid...\n")
            self._get_filepath(files)

    def _get_save_dict(self, agent):
        checkpoint = {'state_size': agent.state_size,
                      'action_size': agent.action_size,
                      'actor_dict': agent.actor.state_dict(),
                      'critic_dict': agent.critic.state_dict()
                      }
        return checkpoint
