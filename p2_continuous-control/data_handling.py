import sys
import numpy as np
import time
from utils import print_bracketing
from argparse import ArgumentParser
import torch
import os.path
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Saver():
    """
    Handles the saving of checkpoints and collection of data to do so. Generates
    the savename and directory for each Agent session.
    PARAMS:
    prefix - usually the name of the framework of the agent being trained, but
            could be manually provided if desired.
    save_dir - this will usually come from a cmdline parser
    file_ext - extension to append to saved weights files. Can be any arbitrary
            string the user desires.
    """
    def __init__(self,
                 prefix,
                 save_dir = 'saves',
                 file_ext = ".agent"):
        """
        Initialize a Saver object.
        """
        self.file_ext = file_ext
        self.save_dir, self.filename = self.generate_savename(prefix, save_dir)

    def load_agent(self, load_file, agent):
        """
        Loads a checkpoint from an earlier trained agent.
        """
        checkpoint = torch.load(load_file, map_location=lambda storage, loc: storage)
        agent.actor.load_state_dict(checkpoint['actor_dict'])
        agent.critic.load_state_dict(checkpoint['critic_dict'])
        agent._hard_update(agent.actor, agent.actor_target)
        agent._hard_update(agent.critic, agent.critic_target)
        statement = "Successfully loaded file: {}".format(load_file)
        print_bracketing(statement)

    def generate_savename(self, prefix, save_dir):
        """
        Generates an automatic savename for training files, will version-up as
        needed.
        """
        base_name = "{}_{}_v".format(prefix, time.strftime("%Y%m%d", time.localtime()))
        files = [f for f in os.listdir(save_dir)]
        files = [f for f in files if base_name in f]
        if len(files)>0:
            ver = [int(re.search("_v(\d+)", file).group(1)) for file in files]
            ver = max(ver) + 1
        else:
            ver = 1
        filename =  "{}{}".format(base_name, ver)
        print_bracketing("Saving to base filename: " + filename)
        save_dir = os.path.join(save_dir, filename)
        return save_dir, filename

    def save_checkpoint(self, agent, save_every):
        """
        Saves the current Agent networks to checkpoint files.
        """

        if agent.episode % save_every:
            return
        save_name = "{}_eps{}_ckpt{}".format(self.filename, agent.episode, self.file_ext)
        full_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = "Saving Agent checkpoint to: {}".format(full_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        self._check_dir(self.save_dir)

        torch.save(self._get_save_dict(agent), full_name)

    def save_final(self, agent):
        """
        Saves a checkpoint after training has finished.
        """

        save_name = "{}_eps{}_FINAL{}".format(self.filename, agent.episode-1, self.file_ext)
        # full_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        full_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = "Saved final Agent weights to: {}".format(full_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        torch.save(self._get_save_dict(agent), full_name)

    def _check_dir(self, dir):
        """
        Creates requested directory if it doesn't yet exist.
        """

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
    def __init__(self,
                 agent=None,
                 args=None,
                 save_dir = '.',
                 log_every = 10):
        if agent==None or args==None:
            print("Blank init for Logger object.")
            return

        self.max_eps = args.num_episodes
        self.quietmode = args.quiet
        self.log_every = log_every
        self.agent_count = agent.agent_count
        self.save_dir = save_dir
        self.log_dir = os.path.join(self.save_dir, 'logs').replace('\\','/')
        self._check_dir(self.log_dir)
        self.filename = os.path.basename(self.save_dir)

        self._init_rewards()
        self._init_logs(self._collect_params(args, agent))

        statement = "Starting training at: {}".format(time.strftime("%H:%M:%S", time.localtime()))
        print_bracketing(statement)
        self.start_time = self.eps_time =  time.time()
        self.scores = []
        self.losses = []

    def graph(self, logdir=None, do_save=True):
        if logdir != None:
            self.log_dir = logdir
            self.filename = os.path.basename(logdir)
            print(self.log_dir)
            for f in os.listdir(self.log_dir):
                if f.endswith("_LOG.txt"):
                    self.paramfile = os.path.join(self.log_dir,f)
                if f.endswith("_actorloss.txt"):
                    self.alossfile = os.path.join(self.log_dir,f)
                if f.endswith("_criticloss.txt"):
                    self.clossfile = os.path.join(self.log_dir,f)
                if f.endswith("_scores.txt"):
                    self.scoresfile = os.path.join(self.log_dir, f)
        self.load_logs()
        self.plot_logs(do_save)


    def load_logs(self):
        """
        Loads data from on-disk log files, for later manipulation and plotting.
        """
        with open(self.scoresfile, 'r') as f:
            self.slines = [float(i) for i in f.read().splitlines()]
        with open(self.alossfile, 'r') as f:
            self.alines = [float(i) for i in f.read().splitlines()]
        with open(self.clossfile, 'r') as f:
            self.clines = [float(i) for i in f.read().splitlines()]
        with open(self.paramfile, 'r') as f:
            loglines = f.read().splitlines()
        pstring = ''
        counter = 0
        params = ['max_steps', 'num_episodes', 'c', 'num_atoms', 'vmin', 'vmax', 'e', 'e_decay', 'e_min', 'gamma', 'actor_learn_rate', 'critic_learn_rate', 'buffer_size', 'batch_size', 'pretrain']
        for line in loglines:
            if line.split(':')[0].lower() in params:
                line += '  '
                counter += len(line)

                if counter > 80:
                    pstring += '\n'
                    counter = 0
                pstring += line
        self.pstring = pstring

    def plot_logs(self, do_save=True):
        """
        Plots data in a matplotlib graph for review and comparison.
        """
        score_x = np.linspace(1, len(self.slines), len(self.slines))
        actor_x = np.linspace(1, len(self.alines), len(self.alines))
        critic_x = np.linspace(1, len(self.clines), len(self.clines))
        dtop = 0.85
        xcount = 5
        xstep = int(len(self.slines)/xcount)
        xticks = np.linspace(0, len(self.slines), xcount, dtype=int)

        fig = plt.figure(figsize=(20,10))
        gs = GridSpec(2, 2, hspace=.5, wspace=.2, top=dtop-0.08)
        ax1 = fig.add_subplot(gs[:,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1])
        gs2 = GridSpec(1,1, bottom=dtop-0.01, top=dtop)
        dummyax = fig.add_subplot(gs2[0,0])
        ax1.plot(score_x, self.slines)
        ax1.set_title("Scores")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score")

        ax2.plot(actor_x, self.alines)
        ax2.set_title("Actor Loss")
        ax2.set_xticks(np.linspace(0, len(self.alines), xcount))
        ax2.set_xticklabels(xticks)
        ax2.set_yticks(np.linspace(min(self.alines), max(self.alines), 5))
        ax2.set_ylabel("Loss", labelpad=10)


        ax3.plot(critic_x, self.clines)
        ax3.set_title("Critic Loss")
        ax3.set_xticks(np.linspace(0, len(self.alines), xcount))
        ax3.set_xticklabels(xticks)
        ax3.set_yticks(np.linspace(min(self.clines), max(self.clines), 5))
        ax3.set_ylabel("Loss", labelpad=20)

        dummyax.set_title(self.pstring, size=13)
        dummyax.axis("off")

        fig.suptitle("Training run {}".format(self.filename), size=40)

        savegraph = os.path.join(self.log_dir, self.filename+"_graph.png")
        if do_save:
            fig.savefig(savegraph)
        else:
            fig.show()
        statement = "Saved graph data to: {}".format(savegraph).replace("\\", "/")
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))


    def log(self, rewards, agent):
        self.rewards += rewards
        self.actor_loss = agent.actor_loss
        self.critic_loss = agent.critic_loss
        if agent.t_step % self.log_every == 0 :
            self._write_losses()


    def _check_dir(self, dir):
        """
        Creates requested directory if it doesn't yet exist.
        """

        if not os.path.isdir(dir):
            os.makedirs(dir)

    def _init_logs(self, params):
        """
        Outputs an initial log of all parameters provided as a list.
        """

        basename = os.path.join(self.log_dir, self.filename)
        self.paramfile = basename + "_LOG.txt"
        self.alossfile = basename + "_actorloss.txt"
        self.clossfile = basename + "_criticloss.txt"
        self.scoresfile = basename + "_scores.txt"
        with open(self.paramfile, 'w') as f:
            for line in params:
                f.write(line + '\n')
        with open(self.alossfile, 'w') as f:
            pass
        with open(self.clossfile, 'w') as f:
            pass
        with open(self.scoresfile, 'w') as f:
            pass
        log_statement = ["Logfiles saved to: {}".format(self.log_dir)]
        log_statement.append("...{}".format(os.path.basename(self.paramfile)))
        log_statement.append("...{}".format(os.path.basename(self.alossfile)))
        log_statement.append("...{}".format(os.path.basename(self.clossfile)))
        log_statement.append("...{}".format(os.path.basename(self.scoresfile)))
        print_bracketing(log_statement)

    def _write_losses(self):
        with open(self.alossfile, 'a') as f:
            f.write(str(self.actor_loss) + '\n')
        with open(self.clossfile, 'a') as f:
            f.write(str(self.critic_loss) + '\n')

    def _write_scores(self, score):
        with open(self.scoresfile, 'a') as f:
            f.write(str(score) + '\n')

    def _collect_params(self, args, agent):
        """
        Creates a list of all the Params used to run this training instance,
        prints this list to the command line if QUIET is not flagged, and stores
        it for later saving to the params log in the saves directory.
        """

        param_list = [self._format_param(arg, args) for arg in vars(args) if arg not in vars(agent)]
        param_list += [self._format_param(arg, agent) for arg in vars(agent)]
        if not self.quietmode: print_bracketing(param_list)
        return param_list

    def _format_param(self, arg, args):
        """
        Formats into PARAM: VALUE for reporting. Strips leading underscores for
        placeholder params where @properties are used for the real value.
        """
        return "{}: {}".format(arg.upper().lstrip("_"), getattr(args, arg))

    def step(self, epsnum):
        epstime, total = self._runtime()
        print("\nEpisode {}/{}... RUNTIME: {}, TOTAL: {}".format(epsnum, self.max_eps, epstime, total))
        self._update_score()
        self._init_rewards()
        print("A LOSS: ", self.actor_loss)
        print("C LOSS: ", self.critic_loss)

    def _runtime(self):
        nowTime = time.time()

        m, s = divmod(nowTime - self.eps_time, 60)
        h, m = divmod(m, 60)
        epstime = "{}h{}m{}s".format(int(h), int(m), int(s))

        m, s = divmod(nowTime - self.start_time, 60)
        h, m = divmod(m, 60)
        total = "{}h{}m{}s".format(int(h), int(m), int(s))

        self.eps_time = nowTime

        return epstime, total

    def _update_score(self):
        score = self.rewards.mean()
        print("{}Return: {}".format("."*10, score))
        self._write_scores(score)
        #self.scores.append(score)

    def _init_rewards(self):
        self.rewards = np.zeros(self.agent_count)

    # def print(self):
    #     # flushlen = len(self.current_log)
    #     # sys.stdout.write(self.current_log)
    #     # sys.stdout.flush()
    #     # sys.stdout.write("\b"*100)
    #     # sys.stdout.flush()
    #     pass
    #
    # def report(self, save_dir):
    #     for detail in self.agent_details:
    #         print(detail)
    #     pass

def gather_args():
    """
    Generate arguments passed from the command line.
    """
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
            help="Run in evalutation mode. Otherwise, will utilize \
                  training mode. In default EVAL mode, NUM_EPISODES is set \
                  to 1 and MAX_STEPS to 1000.",
            action="store_true")
    parser.add_argument("-feval", "--force_eval",
            help="Force evaluation mode to run with specified NUM_EPISODES \
                  and MAX_STEPS param.",
            action="store_true")
    parser.add_argument("-gamma",
            help="Gamma (Discount rate).",
            type=float,
            default=0.99)
    parser.add_argument("-max", "--max_steps",
            help="How many timesteps to explore each episode, if a \
                  Terminal state is not reached first",
            type=int,
            default=1000)
    parser.add_argument("--nographics",
            help="Run Unity environment without graphics displayed.",
            action="store_true")
    parser.add_argument("-num", "--num_episodes",
            help="How many episodes to train?",
            type=int,
            default=200)
    parser.add_argument("-pre", "--pretrain",
            help="How many trajectories to randomly sample into the \
                  ReplayBuffer before training begins.",
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
            help="Use this flag to automatically use the latest save file \
                  to run in DEMO mode (instead of choosing from a prompt).",
            action="store_true")
    parser.add_argument("-file", "--filename",
            help="Path agent weights file to load. ",
            type=str,
            default=None)
    parser.add_argument("-savedir", "--save_dir",
            help="Directory to find saved agent weights.",
            type=str,
            default="saves")
    args = parser.parse_args()

    ############################################################################
    #             PROCESS ARGS AFTER COMMAND LINE GATHERING                    #

    # Pretrain length can't be less than batch_size
    assert args.pretrain >= args.batch_size, "PRETRAIN less than BATCHSIZE."
    # Always use GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Limit the length of evaluation runs unless user forces cmdline args
    if args.eval and not args.force_eval:
        args.num_episodes = 1
        args.max_steps = 1000

    # Determine whether to load a file, and if so, set the filename
    args.load_file = _get_agent_file(args)

    return args



def _get_agent_file(args):
    """
    Checks to see what sort of loading, if any, to do.
    Returns one of:
        -FILENAME... if flagged with a specific filename on the cmdline
        -LASTEST FILE... if flagged to load the most recently saved weights
        -USER FILE... a user selected file from a list prompt
        -FALSE... if no loading is needed, return false and skip loading
    """

    invalid_filename = "Requested filename is invalid."
    no_files_found = "Could not find any files in: {}".format(args.save_dir)
    if args.resume or args.eval:
        if args.filename is not None:
            assert os.path.isfile(args.filename), invalid_filename
            return args.filename
        files = _get_files(args.save_dir)
        assert len(files) > 0, no_files_found
        if args.latest:
            return files[-1]
        else:
            return _get_filepath(files)
    else:
        return False



def _get_files(save_dir):
    """
    Returns a list of files in a given directory, sorted by last-modified.
    """

    file_list = []
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file.endswith(".agent"):
                file_list.append(os.path.join(root, file))
    return sorted(file_list, key=lambda x: os.path.getmtime(x))



def _get_filepath(self, files):
    """
    Prompts the user about what save to load, or uses the last modified save.
    """

    load_file_prompt = " (LATEST)\n\nPlease choose a saved Agent training file (or: q/quit): "
    user_quit_message = "User quit process before loading a file."
    message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
    message = '\n'.join(message).replace('\\', '/')
    message = message + load_file_prompt
    save_file = input(message)
    if save_file.lower() in ("q", "quit"):
        raise KeyboardInterrupt(user_quit_message)
    try:
        file_index = len(files) - int(save_file)
        assert file_index >= 0
        return files[file_index]
    except:
        print("")
        print_bracketing('Input "{}" is INVALID...'.format(save_file))
        _get_filepath(files)
