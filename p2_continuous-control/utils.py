import os.path
import re
import time

import torch



def print_bracketing(info=None, do_upper=True, do_lower=True):
    mult = 50
    if type(info) is not list and info is not None:
        mult = max(mult, len(info))
        info = [info]
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    if do_upper: print(upper)
    if info is not None:
        for line in info:
            print(line.center(mult))
    if do_lower: print(lower)

class Saver:
    def __init__(self, agent, args):
        self.file_ext = ".agent"
        self.save_dir = args.save_dir

        self.filename = self.generate_savename(agent.framework)
        self._check_dir(os.path.join(self.save_dir, self.filename))
        self._write_init_log(agent)
        print_bracketing("Saving to base filename: " + self.filename)


    def generate_savename(self, agent_name):
        """Generates an automatic savename for training files, will version-up as
           needed.
        """
        t = time.localtime()
        savename = "{}_{}_v".format(agent_name, time.strftime("%Y%m%d", time.localtime()))
        files = [f for f in os.listdir(self.save_dir)]
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
        torch.save(self._get_save_dict(agent), full_name)

    def save_final(self, agent):
        """
        Saves a checkpoint after training has finished.
        """
        save_name = "{}_eps{}_FINAL{}".format(self.filename, agent.episode-1, self.file_ext)
        full_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = "Saved final Agent weights to: {}".format(full_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        torch.save(self._get_save_dict(agent), full_name)

    def _check_dir(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)

    def _get_save_dict(self, agent):
        checkpoint = {'state_size': agent.state_size,
                      'action_size': agent.action_size,
                      'actor_dict': agent.actor.state_dict(),
                      'critic_dict': agent.critic.state_dict()
                      }
        return checkpoint

    def _write_init_log(self, agent):
        """
        """
        file = os.path.join(self.save_dir, self.filename, self.filename) + "_LOG.txt"
        with open(file, 'w') as f:
            for arg in vars(agent):
                f.write("{}: {}\n".format(arg.upper(), getattr(agent, arg)))
        print_bracketing("Logfile saved to: {}".format(file))
