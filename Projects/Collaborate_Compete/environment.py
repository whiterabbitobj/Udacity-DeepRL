# -*- coding: utf-8 -*-
import torch
from unityagents import UnityEnvironment
# from mlagents.envs import UnityEnvironment
from utils import print_bracketing
import platform

class Environment:
    """
    Wrapper for the Udacity Tennis environment. Keeps the
    main body code a bit more neat and allows for easier access to certain
    params elsewhere.
    """
    def __init__(self, args, id=0):
        """
        Initialize an environment wrapper.
        """

        self.train = not args.eval

        print("LOADING ON SYSTEM: {}".format(platform.system()))

        print_bracketing(do_lower=False)
        if platform.system() == 'Linux':
            unity_filename = "Tennis_Linux_NoVis/Tennis.x86_64"
        elif platform.system() == 'Darwin':
            print("MacOS not supported in this code!")
        else:
            unity_filename = "Tennis_Windows_x86_64/Tennis.exe"
        self.env = UnityEnvironment(file_name=unity_filename, worker_id=id, no_graphics=args.nographics)
        print_bracketing(do_upper=False)

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Environment resets itself when the class is instantiated
        self.reset()

        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.state.shape[1]
        self.agent_count = len(self.env_info.agents)

    def reset(self):
        """
        Resets the environment.
        """

        self.env_info = self.env.reset(train_mode = self.train)[self.brain_name]

    def close(self):
        """
        Closes the environment when Agent is done interacting with it.
        """

        self.env.close()

    def step(self, action):
        """
        Returns REWARDS, NEXT_STATES, DONES based on the actions provided.
        """

        self.env_info = self.env.step(action)[self.brain_name]
        next_state = self.state
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return next_state, reward, done

    @property
    def state(self):
        """
        Returns the STATES as a tensor.
        """
        states = self.env_info.vector_observations[0]
        return torch.from_numpy(states).float().unsqueeze(0)
