import os.path
import torch
from argparse import ArgumentParser
from utils import print_bracketing
from data_handling import Loader, Saver, gather_args

class Meta():
    def __init__(self):
        """
        Initialization wrapper for Deep Reinforcement Learning tasks.
        """
        # self.args = gather_args()
        # self.saver = Saver()
        # self.logger = Logger()
        self.quietmode = self.args.quiet
        # self._collect_print_statements()
        # self.load_file = self._get_agent_file(self.args)

    #
    # def init_session(self, env, agent):
    #     self.logger.init(agent, self.args)
    #
    # def init_training(self, agent):
    #     self.saver.init(agent, self.args)
    #
    # def init_eval(self, agent):
    #     self.logger.init(agent, self.args)


    #
    # def _get_agent_file(self, args):
    #     """
    #     Checks to see what sort of loading, if any, to do.
    #     Returns one of:
    #         -FILENAME... if flagged with a specific filename on the cmdline
    #         -LASTEST FILE... if flagged to load the most recently saved weights
    #         -USER FILE... a user selected file from a list prompt
    #         -FALSE... if no loading is needed, return false and skip loading
    #     """
    #
    #     if args.resume or args.eval:
    #         if args.filename is not None:
    #             assert os.path.isfile(args.filename), self.invalid_filename
    #             return args.filename
    #         files = self._get_files(args.save_dir)
    #         assert len(files) > 0, self.no_files_found
    #         if args.latest:
    #             return files[-1]
    #         else:
    #             return self._get_filepath(files)
    #     else:
    #         return False
    #
    # def _get_files(self, save_dir):
    #     """
    #     Returns a list of files in a given directory, sorted by last-modified.
    #     """
    #
    #     file_list = []
    #     for root, _, files in os.walk(save_dir):
    #         for file in files:
    #             file_list.append(os.path.join(root, file))
    #     return sorted(file_list, key=lambda x: os.path.getmtime(x))
    #
    # def _get_filepath(self, files):
    #     """
    #     Prompts the user about what save to load, or uses the last modified save.
    #     """
    #
    #     message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
    #     message = '\n'.join(message).replace('\\', '/')
    #     message = message + self.load_file_prompt
    #     save_file = input(message)
    #     if save_file.lower() in ("q", "quit"):
    #         raise KeyboardInterrupt(self.user_quit_message)
    #     try:
    #         file_index = len(files) - int(save_file)
    #         assert file_index >= 0
    #         return files[file_index]
    #     except:
    #         print("")
    #         print_bracketing('Input "{}" is INVALID...'.format(save_file))
    #         self._get_filepath(files)
