import os.path
import time
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from unityagents import UnityEnvironment
from PIL import Image


# def anneal_parameter(param, anneal_rate, param_min):
#     return min(param * anneal_rate, param_min)


##########
## Interact with the environment
##########

# def process_frame(state):
#     #state = torch.from_numpy(state.squeeze(0).astype(np.float32).transpose(2,0,1)) #[3,84,84]
#     state = state.squeeze(0).transpose(2,0,1)
#     #return red channel & crop frame
#     state = state[0,5:-40,5:-5]
#     state = np.ascontiguousarray(state, dtype=np.float32)
#     state = torch.from_numpy(state).unsqueeze(0)
#     return state
#     #return state[0,5:-40,5:-5].unsqueeze(0) #[1,39,74] crop image to save calculation time



def load_environment(args, frame_buffer):
    if args.pixels:
        unity_filename = "VisualBanana_Windows_x86_64/Banana.exe"
    else:
        unity_filename = "Banana_Windows_x86_64/Banana.exe"
    env = UnityEnvironment(file_name=unity_filename, no_graphics=args.nographics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=args.train)[brain_name]
    nA = brain.vector_action_space_size
    if args.pixels:
        state = env_info.visual_observations[0]
        state_size = list(frame_buffer.process_frame(state).shape)
        state_size[0] = args.framestack
    else:
        state_size = len(env_info.vector_observations[0])
    return env, env_info, brain_name, nA, state_size



def get_state(env_info, agent, done):
    if agent.pixels:
        state = env_info.visual_observations[0]
        agent.memory.stack(process_frame(state), done)
        return agent.memory.get_stack().unsqueeze(0)
    else:
        state = env_info.vector_observations[0]
        state =  torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
    return state



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



def save_checkpoint(agent, scores, args):
    """Saves the current Agent's learning dict as well as important parameters
       involved in the latest training.
    """
    if not args.train:
        return

    agent.q.to('cpu')
    checkpoint = {'agent_type': agent.framework,
                  'state_size': agent.nS,
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



def print_verbose_info(agent, env_info, args):
    """
    Prints extra data if --verbose flag is set.
    """
    if not args.verbose:
        return

    print("ARGS:\n", "-"*5)
    for arg in vars(args):
        if arg == "sep": continue
        print("{}: {}".format(arg.upper(), getattr(args, arg)))
    print(args.sep, "\nVARS:\n", "-"*5)
    print("Device: ", agent.device)
    print("Action Size: ", agent.nA)
    print("Processed state looks like: ", agent.nS)
    print('Number of agents: ', len(env_info.agents))
    print("Number of Episodes: ", args.num_episodes)
    print("{1}\n{0}\n{1}".format(agent.q, args.sep))



def print_status(i_episode, scores, agent, args):
    if i_episode % args.print_every == 0:
        print("\nEpisode {}/{}, avg score for last {} episodes: {:3f}".format(
                i_episode, args.num_episodes, args.print_every, np.mean(scores[-args.print_every:])))
        if args.verbose:
            print("Epsilon: {}\n".format(agent.epsilon))



def get_runtime(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    return  "{}h{}m{}s".format(int(h), int(m), int(s))
