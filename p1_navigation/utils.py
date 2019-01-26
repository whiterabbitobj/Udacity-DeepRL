import os.path
import matplotlib.pyplot as plt
import numpy as np

import torch
from agent import DQN_Agent



def load_checkpoint(filepath, device, args):
    '''Loads a checkpoint from an earlier trained agent.
    '''
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    if checkpoint['agent_type'] == 'DQN':
        agent = DQN_Agent(checkpoint['state_size'], checkpoint['action_size'], device, args)
    agent.qnet_local.load_state_dict(checkpoint['state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])

    return agent



def load_filepath(use_latest):
    separator = "#"*50 + "\n"
    files = [str(f) for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1] == '.pth']
    files = sorted(files, key=lambda x: os.path.getmtime(x))
    if use_latest:
        save_file = files[-1]
    else:
        message = separator + '\n'.join(["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]) + " (LATEST)\nPlease choose a saved Agent training file: "
        save_file = input(message)
        try:
            file_index = len(files) - int(save_file)
            if file_index < 0:
                raise Exception()
            save_file = files[file_index]
        except:
            print("Invalid choice.")
            load_filepath(use_latest)
    print("{0}Proceeding with file: {1}\n{0}".format(separator, save_file))
    return save_file



def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    return