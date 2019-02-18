import os.path
import time
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from unityagents import UnityEnvironment
#from PIL import Image
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

        if self.pixels:
            state = self.env_info.visual_observations[0]
            self.state_size = list(self.process_frame(state).shape)
            self.state_size[0] = args.framestack
        else:
            self.state_size = len(self.env_info.vector_observations[0])

    def process_frame(self, state): #GRAYSCALE IMPLEMENTATION
        frame = state.squeeze(0) * 255 #uncompress data from 0-1 to 0-255
        frame = frame[35:-2,2:-2,:] #crop frame
        frame = np.ascontiguousarray(frame, dtype=np.uint8) #ensure cropped data is not kept in memory
        transforms = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
        frame = transforms(frame)
        # plt.imshow(T.ToPILImage()(frame))
        # plt.show()
        return frame #add dimension for stacking

    # def process_frame(self, state): #RED CHANNEL IMPLEMENTATION
    #     frame = state.squeeze(0).transpose(2,0,1) #remove banana env extra dimension & transpose to tensor style encoding
    #     frame = frame[0,5:-40,5:-5] #return red channel & crop frame
    #     frame = np.ascontiguousarray(frame, dtype=np.float32)# / 255 #ensure cropped data is not kept in memory
    #     return torch.from_numpy(frame).unsqueeze(0) #add dimension for stacking

    # def process_color_frame(self, state):
    #     frame = state.transpose(0,3,1,2) #remove banana env extra dimension & transpose to tensor style encoding
    #     frame = frame[:,:,5:-40,5:-5] #return red channel & crop frame
    #     frame = np.ascontiguousarray(frame, dtype=np.float32) / 255 #ensure cropped data is not kept in memory
    #     return torch.from_numpy(frame).unsqueeze(0) #add dimension for stacking

    def stack(self, frame, reset):
        if reset:
            self.phi = deque([frame for i in range(self.phi.maxlen)], maxlen=self.phi.maxlen)
        else:
            self.phi.append(frame)
        return

    def get_stack(self):
        return torch.cat(tuple(self.phi),dim=0).to(self.device)

    def state(self, reset=False):
        if self.pixels:
            state = self.env_info.visual_observations[0]
            frame = self.process_frame(state)
            self.stack(frame, reset)
            return self.get_stack().unsqueeze(0)
        else:
            state = self.env_info.vector_observations[0]
            return torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return reward, done

    def reset(self):
        self.env_info = self.env.reset(train_mode=self.training)[self.brain_name]

    def close(self):
        self.env.close()



##########
## Saving & Loading
##########

def generate_savename(agent_name, scores, print_every):
    """Generates an automatic savename for training files, will version-up as
       needed.
    """
    savename = "{}_{}_v".format(agent_name, time.strftime("%Y%m%d", time.gmtime()))

    files = [f for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1] == '.pth']
    files = [f for f in files if savename in f]
    if len(files)>0:
        ver = [int(re.search("_v(\d+)", file).group(1)) for file in files]
        ver = max(ver) + 1
    else:
        ver = 1
    eps = len(scores)
    avg_score = np.mean(scores[-print_every:])
    return "{}{}_{}eps_{:.2f}score{}".format(savename, ver, eps, avg_score, ".pth")



def save_checkpoint(agent, scores, args, state_size):
    """Saves the current Agent's learning dict as well as important parameters
       involved in the latest training.
    """
    if not args.train:
        return

    agent.q.to('cpu')
    checkpoint = {'agent_type': agent.framework,
                  'state_size': state_size,
                  'action_size': agent.nA,
                  'state_dict': agent.q.state_dict(),
                  'optimizer': agent.optimizer.state_dict(),
                  'scores': scores
                  }
    save_name = generate_savename(agent.framework, scores, args.print_every)
    torch.save(checkpoint, save_name)
    print("{}\nSaved agent data to: {}".format("#"*50, save_name))



def load_checkpoint(filepath, args):
    """Loads a checkpoint from an earlier trained agent.
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



def load_filepath(sep):
    """Prompts the user about what save to load, or uses the last modified save.
    """
    files = [str(f) for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1] == '.pth']
    if len(files) == 0:
        print("Oops! Couldn't find any save files in the current directory.")
        return None

    files = sorted(files, key=lambda x: os.path.getmtime(x))
    if args.latest:
        print("{0}Proceeding with file: {1}\n{0}".format(sep, files[-1]))
        return files[-1]
    else:
        message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
        message = '\n'.join(message)
        message = sep + message + " (LATEST)\n\nPlease choose a saved Agent training file (or: q/quit): "
        save_file = input(message)
        if save_file.lower() == "q" or save_file.lower() == "quit":
            print("Quit before loading a file.")
            return None
        try:
            file_index = len(files) - int(save_file)
            if file_index < 0:
                raise Exception()
            save_file = files[file_index]
            print("{0}\nProceeding with file: {1}\n{0}".format(sep, save_file))
            return save_file
        except:
            print("\nInput invalid...\n")
            load_filepath()

##########
## Print utilities
##########

def report_results(scores, start_time):
    """
    Prints runtime.
    Displays a simple graph of training data, score per episode across all episodes.
    """
    print("TOTAL RUNTIME: {}.".format(get_runtime(start_time)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()



def print_verbose_info(agent, args, state_size):
    """
    Prints extra data if --verbose flag is set.
    """
    if not args.verbose:
        return

    print("ARGS:")
    print("-"*5)
    for arg in vars(args):
        if arg == "sep": continue
        print("{}: {}".format(arg.upper(), getattr(args, arg)))
    print(args.sep, "\nVARS:")
    print("-"*5)
    print("Device: ", agent.device)
    print("Action Size: ", agent.nA)
    print("Processed state looks like: ", state_size)
    print("Number of Episodes: ", args.num_episodes)
    print("{1}\n{0}\n{1}".format(agent.q, args.sep))



def print_status(i_episode, scores, agent, args):
    if i_episode % args.print_every == 0:
        print("\nEpisode {}/{}, avg score for last {} episodes: {:3f}".format(
                i_episode, args.num_episodes, args.print_every, np.mean(scores[-args.print_every:])))
        if args.verbose:
            print("Epsilon: {}".format(agent.epsilon))
            print("Timesteps: ", agent.t_step, "\n\n")



def get_runtime(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    return  "{}h{}m{}s".format(int(h), int(m), int(s))
