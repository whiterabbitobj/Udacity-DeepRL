#import Logger
#import Saver
import os.path
# import re
# import time

import torch
from argparse import ArgumentParser
from utils import print_bracketing


class Meta():
    def __init__(self):
        self.args = self._get_args()
        self._process_args(self.args)
        self.user_quit_message = "User quit process before loading a file."
        self.no_files_found = "Could not find any files in: {}".format(self.args.save_dir)

        self.load_file = self._get_agent_file(self.args)

        pass

    def load_agent(self, agent):
        """
        Loads a checkpoint from an earlier trained agent.
        """
        file = self.load_file
        checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
        agent.actor.load_state_dict(checkpoint['actor_dict'])
        agent.critic.load_state_dict(checkpoint['critic_dict'])
        agent._hard_update(agent.actor, agent.actor_target)
        agent._hard_update(agent.critic, agent.critic_target)
        statement = "Successfully loaded file: {}".format(file)
        print_bracketing(statement)

    def _get_agent_file(self, args):
        if args.resume or args.eval:
            if args.filename is not None:
                assert os.path.isfile(args.filename), "Requested filename is invalid."
                return args.filename
            files = self._get_files(args.save_dir)
            assert len(files) > 0, self.no_files_found
            if args.latest:
                return files[-1]
            else:
                return self._get_filepath(files)
        else:
            return False

    def _get_files(self, save_dir):
        file_list = []
        for root, _, files in os.walk(save_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return sorted(file_list, key=lambda x: os.path.getmtime(x))

    def _get_filepath(self, files):
        """
        Prompts the user about what save to load, or uses the last modified save.
        """
        prompt = " (LATEST)\n\nPlease choose a saved Agent training file (or: q/quit): "
        message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
        message = '\n'.join(message).replace('\\', '/')
        message = message + prompt
        save_file = input(message)
        if save_file.lower() in ("q", "quit"):
            raise KeyboardInterrupt(self.user_quit_message)
        try:
            file_index = len(files) - int(save_file)
            assert file_index >= 0
            return files[file_index]
        except:
            print("")
            print_bracketing('Input "{}" is INVALID...'.format(save_file))
            self._get_filepath(files)

    def _process_args(self, args):
        """
        Take the command line arguments and process them as needs require.
        """

        # Pretrain length can't be less than batch_size
        assert args.pretrain >= args.batch_size, "PRETRAIN less than BATCHSIZE."
        # Always use GPU if available
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Don't allow the user to run EVAL mode for more than 10 episodes
        if args.eval and args.num_episodes > 10:
            print("In eval mode, num_episodes is set to not more than 10.")
            args.num_episodes = 10
        # Default to printing all the ARGS info to the command line for review
        if not args.quiet:
            arg_print = []
            for arg in vars(args):
                if arg == "quiet": continue
                arg_statement = "{}: {}".format(arg.upper(), getattr(args, arg))
                arg_print.append(arg_statement)
            print_bracketing(arg_print)

    def _get_args(self):
        parser = ArgumentParser(description="Continuous control environment for Udacity DeepRL course.",
                usage="")

        parser.add_argument("-alr", "--actor_learn_rate",
                help="Alpha (Learning Rate).",
                type=float,
                default=1e-3)
        parser.add_argument("-clr", "--critic_learn_rate",
                help="Alpha (Learning Rate).",
                type=float,
                default=1e-4)
        parser.add_argument("-bs", "--batch_size",
                help="Size of each batch between learning updates",
                type=int,
                default=128)
        parser.add_argument("-buffer", "--buffer_size",
                help="How many past timesteps to keep in memory.",
                type=int,
                default=300000)
        parser.add_argument("-C", "--C",
                help="How many timesteps between hard network updates.",
                type=int,
                default=1000)
        parser.add_argument("-eval", "--eval",
                help="Run in evalutation mode. Otherwise, will utilize training mode.",
                action="store_true")
        parser.add_argument("-gamma",
                help="Gamma (Discount rate).",
                type=float,
                default=0.99)
        parser.add_argument("-max", "--max_steps",
                help="How many timesteps to explore each episode, if a Terminal \
                      state is not reached first",
                type=int,
                default=1000)
        parser.add_argument("--nographics",
                help="Run Unity environment without graphics displayed.",
                action="store_true")
        parser.add_argument("-num", "--num_episodes",
                help="How many episodes to train?",
                type=int,
                default=100)
        parser.add_argument("-pre", "--pretrain",
                help="How many trajectories to randomly sample into the ReplayBuffer\
                      before training begins.",
                type=int,
                default=5000)
        parser.add_argument("--quiet",
                help="Print less while running the agent.",
                action="store_true")
        parser.add_argument("--resume",
                help="Resume training from a checkpoint.",
                action="store_true")
        parser.add_argument("-roll", "--rollout",
                help="How many experiences to use in N-Step returns",
                type=int,
                default=5)
        parser.add_argument("-se", "--save_every",
                help="How many episodes between saves.",
                type=int,
                default=10)
        parser.add_argument("-t", "--tau",
                help="Soft network update weighting.",
                type=float,
                default=0.0005)
        parser.add_argument("--latest",
                help="Use this flag to automatically use the latest save file to \
                      run in DEMO mode (instead of choosing from a prompt).",
                action="store_true")
        parser.add_argument("-file", "--filename",
                help="Path agent weights file to load. ",
                type=str,
                default=None)
        parser.add_argument("-savedir", "--save_dir",
                help="Directory to find saved agent weights.",
                type=str,
                default="saves")
        return parser.parse_args()
