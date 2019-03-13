import sys
import numpy as np
import time
from utils import print_bracketing

class Saver:
    def __init__(self, agent, args):
        self.file_ext = ".agent"
        self.save_dir = args.save_dir
        self.filename = self.generate_savename(agent.framework)
        self._check_dir(os.path.join(self.save_dir, self.filename))
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


class Logger:
    def __init__(self, agent, args):
        self.max_eps = args.num_episodes
        self.current_log = ''
        self.full_log = ''
        self.agent_count = agent.agent_count
        self.scores = []
        self.save_dir = args.save_dir


        self._write_init_log(self._collect_params(args, agent))
        self._reset_rewards()

    def _write_init_log(self, params):
        """
        Outputs an initial log of all parameters provided as a list.
        """
        ext = "_LOG.txt"
        file = os.path.join(self.save_dir, self.filename, self.filename) + ext
        with open(file, 'w') as f:
            for line in params:
                f.write(line)
        print_bracketing("Logfile saved to: {}".format(file))

    def _collect_params(self, args, agent):
        """
        Creates a list of all the Params used to run this training instance,
        prints this list to the command line if QUIET is not flagged, and stores
        it for later saving to the params log in the saves directory.
        """
        # Default to printing all the ARGS info to the command line for review
        param_list = [self._format_param(arg, self.args) for arg in vars(self.args) if arg not in vars(agent)]
        param_list.append("\n")
        param_list += [self._format_param(arg, agent) for arg in vars(agent)]
        if not self.quietmode: print_bracketing(param_list)
        return param_list

    # def add(self, log):
    #     self.current_log += str(log)

    def start_clock(self):
        t = time.localtime()
        statement = "Starting training at: {}".format(time.strftime("%H:%M:%S", time.localtime()))
        print_bracketing(statement)
        self.start_time = time.time()

    def step(self, eps):

        print("\nEpisode {}/{}... RUNTIME: {}".format(eps, self.max_eps, self._runtime()))
        self._update_score()
        self._reset_rewards()

    def _runtime(self):
        m, s = divmod(time.time() - self.start_time, 60)
        h, m = divmod(m, 60)
        return "{}h{}m{}s".format(int(h), int(m), int(s))

    def _update_score(self):
        score = self.rewards.mean()
        print("{}Return: {}".format("."*10, score))
        self.scores.append(score)

    def _reset_rewards(self):
        self.rewards = np.zeros(self.agent_count)

    def print(self):
        # flushlen = len(self.current_log)
        # sys.stdout.write(self.current_log)
        # sys.stdout.flush()
        # sys.stdout.write("\b"*100)
        # sys.stdout.flush()
        pass

    def report(self, save_dir):
        for detail in self.agent_details:
            print(detail)
        pass
