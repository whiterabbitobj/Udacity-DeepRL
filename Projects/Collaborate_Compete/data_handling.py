# -*- coding: utf-8 -*-
import sys
import numpy as np
import time
from utils import print_bracketing, check_dir
from argparse import ArgumentParser
import torch
import os.path
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque

class Saver():
    """
    Handles the saving of checkpoints and collection of data to do so. Generates
    the savename and directory for each Agent session.
    PARAMS:
    prefix - usually the name of the framework of the agent being trained, but
            could be manually provided if desired.
    agent - agent to either load or save data to/from.
    save_dir - this will usually come from a cmdline parser
    load_file - filename of saved weights to load into the current agent.
    file_ext - extension to append to saved weights files. Can be any arbitrary
            string the user desires.
    """
    def __init__(self,
                 prefix,
                 agent,
                 save_dir = 'saves',
                 load_file = None,
                 file_ext = ".agent"
                 ):
        """
        Initialize a Saver object.
        """

        self.file_ext = file_ext
        self.save_dir, self.filename = self.generate_savename(prefix, save_dir)
        if load_file:
            self._load_agent(load_file, agent)
        else:
            statement = "Saving to base filename: {}".format(self.filename)
            print_bracketing(statement)

    def generate_savename(self, prefix, save_dir):
        """
        Generates an automatic savename for training files, will version-up as
        needed.
        """
        check_dir(save_dir)
        timestamp = time.strftime("%Y%m%d", time.localtime())
        base_name = "{}_{}_v".format(prefix, timestamp)
        files = [f for f in os.listdir(save_dir)]
        files = [f for f in files if base_name in f]
        if len(files)>0:
            ver = [int(re.search("_v(\d+)", file).group(1)) for file in files]
            ver = max(ver) + 1
        else:
            ver = 1
        filename =  "{}{:03d}".format(base_name, ver)
        save_dir = os.path.join(save_dir, filename)
        return save_dir, filename

    def save_checkpoint(self, agent, save_every):
        """
        Preps a checkpoint save file at intervals controlled by SAVE_EVERY.
        """
        if not agent.episode % save_every == 0:
            return
        mssg = "Saving Agent checkpoint to: "
        save_name = "{}_eps{:04d}_ckpt".format(self.filename, agent.episode)
        self._save(agent, save_name, mssg)

    def save_final(self, agent):
        """
        Preps a final savefile after training has finished.
        """
        mssg = "Saved final Agent weights to: "
        save_name = "{}_eps{:04d}_FINAL".format(self.filename, agent.episode-1)
        self._save(agent, save_name, mssg)

    def _save(self, multi_agent, save_name, mssg):
        """
        Does the actual saving bit.
        """

        basename = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = mssg + basename + "..."
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        check_dir(self.save_dir)
        for idx, agent in enumerate(multi_agent.agents):
            suffix =  "_agent{}".format(idx+1) + self.file_ext
            torch.save(self._get_save_dict(agent), basename + suffix)

    def _get_save_dict(self, agent):
        """
        Prep a dictionary of data from the current Agent.
        """

        checkpoint = {'actor_dict': agent.actor.state_dict(),
                      'critic_dict': agent.critic.state_dict()
                      }
        return checkpoint

    def _load_agent(self, load_file, agent):
        """
        Loads a checkpoint from an earlier trained agent.
        """

        checkpoint = torch.load(load_file, map_location=lambda storage, loc: storage)
        agent.q.load_state_dict(checkpoint['q_dict'])
        agent._hard_update(agent.q, agent.q_target)
        statement = "Successfully loaded file: {}".format(load_file)
        print_bracketing(statement)


class Logger:
    """
    Handles logging training data and printing to log files. Creates a graph at
    the end of training to compare data in a nice format. Log files are stored
    so the data can also be used elsewhere as needed. Initializing a blank
    Logger object allows to manually provide a log directory from which to parse
    data and construct a graph. This is very useful if training is still running
    but one wants to utilize a Jupyter Notebook to monitor current results.
    PARAMS:
    agent - Logger collects the params of both ARGS and AGENT in order to log
        the training session details.
    args - Logger collects the params of both ARGS and AGENT in order to log
        the training session details.
    save_dir - directory where current session saves are being stored. Logger
        will create a /logs/ directory here for storing data.
    log_every - how many timesteps between each logging of losses. Scores are
        logged every episode.
    """
    def __init__(self,
                 multi_agent=None,
                 args=None,
                 save_dir = '.',
                 log_every = 50, #timesteps
                 print_every = 5 #episodes
                 ):
        """
        Initialize a Logger object.
        """
        self.save_dir = save_dir
        if multi_agent==None or args==None:
            print("Blank init for Logger object. Most functionality limited.")
            return
        self.eval = args.eval
        self.framework =  multi_agent.framework
        self.max_eps = args.num_episodes
        self.quietmode = args.quiet
        self.log_every = log_every
        self.agent_count =  multi_agent.agent_count
        self.log_dir = os.path.join(self.save_dir, 'logs').replace('\\','/')
        self.filename = os.path.basename(self.save_dir)
        self.logs_basename = os.path.join(self.log_dir, self.filename)
        self.start_time = self.prev_timestamp =  time.time()
        self.scores = deque(maxlen=print_every)
        self.print_every = print_every
        self._init_rewards()
        if not self.eval:

            timestamp = time.strftime("%H:%M:%S", time.localtime())
            statement = "Starting training at: {}".format(timestamp)
            print_bracketing(statement)

            check_dir(self.log_dir)
            params = self._collect_params(args,  multi_agent)

            self._init_logs(params, multi_agent)

    def log(self, rewards, multi_agent):
        """
        After each timestep, keep track of loss and reward data.
        """

        self.rewards += rewards
        if self.eval:
            return

        # self.loss = agent.loss
        # Writes the loss data to an on-disk logfile every LOG_EVERY timesteps
        if multi_agent.t_step % self.log_every == 0:
            self._write_losses(multi_agent)

    def step(self, eps_num, multi_agent):
        """
        After each episode, report data on runtime and score. If not in
        QUIETMODE, then also report the most recent losses.
        """

        self.scores.append(self._update_score())
        self._init_rewards()

        if self.eval:
            print("Score: {}".format(self.scores[-1]))
            return

        if eps_num % self.print_every == 0:
            eps_time, total_time = self._runtime()
            print("\nEpisode {}/{}... Runtime: {}, Total: {}".format(eps_num, self.max_eps, eps_time, total_time))
            if not self.quietmode:
                for idx, agent in enumerate(multi_agent.agents):
                    print("Agent {}... actorloss: {:5f}, criticloss: {:5f}".format(idx, agent.actor_loss, agent.critic_loss))
                # print("Epsilon: {:6f}, Loss: {:6f}".format(epsilon, self.loss))
            print("{}Avg return over previous {} episodes: {:5f}\n".format("."*5, self.print_every, np.array(self.scores).mean()))

    def load_logs(self):
        """
        Loads data from on-disk log files, for later manipulation and plotting.
        """

        with open(self.scoresfile, 'r') as f:
            self.slines = np.array([float(i) for i in f.read().splitlines()])
        with open(self.netlossfile, 'r') as f:
            self.nlines = np.array([float(i) for i in f.read().splitlines()])
        with open(self.paramfile, 'r') as f:
            loglines = f.read().splitlines()

        # List of the desired params to print on the graph for later review
        params_to_print = ['max_steps', 'num_episodes', 'c', 'num_atoms',
            'vmin', 'vmax', 'epsilon', 'epsilon_decay', 'epsilon_min', 'gamma',
            'learn_rate', 'buffer_size', 'batch_size', 'pretrain', 'rollout',
            'l2_decay']

        sess_params = ''
        counter = 0
        for line in loglines:
            if line.split(':')[0].lower() in params_to_print:
                line += '  '
                counter += len(line)
                if counter > 80:
                    sess_params += '\n'
                    counter = 0
                sess_params += line
        self.sess_params = sess_params

    def _moving_avg(self, data, avg_across):
        """
        Returns a curve that is the average of a noisier curve.
        """

        avg_across = int(avg_across)
        window = np.ones(avg_across)/avg_across
        data = np.pad(data, avg_across, mode="mean", stat_length=10)
        return np.convolve(data, window, 'same')[avg_across:-avg_across]

    def plot_logs(self, save_to_disk=True):
        """
        Plots data in a matplotlib graph for review and comparison.
        """

        score_x = np.linspace(1, len(self.slines), len(self.slines))
        loss_x = np.linspace(1, len(self.nlines), len(self.nlines))
        # critic_x = np.linspace(1, len(self.nlines), len(self.nlines))
        dtop = 0.85
        xcount = 5
        bg_color = 0.925
        ma100_color = (1, .2, .3)
        ma200_color = (.38,1,.55)
        xstep = int(len(self.slines)/xcount)
        xticks = np.linspace(0, len(self.slines), xcount, dtype=int)
        n_yticks = np.linspace(min(self.nlines), max(self.nlines), 5)
        score_window = min(100, len(self.slines))
        nlines_ratio = len(self.nlines)/len(self.slines)
        annotate_props = dict(facecolor=(0.1,0.3,0.5), alpha=0.85, edgecolor=(0.2,0.3,0.6), linewidth=2)

        score_mean = self.slines[-score_window:].mean()
        score_std = self.slines[-score_window:].std()
        score_report = "{0}eps MA score: {1:.2f}\n{0}eps STD: {2:.3f}".format(score_window, score_mean, score_std)

        loss_mean = self.nlines[-int(score_window*nlines_ratio):].mean()
        loss_std = self.nlines[-int(score_window*nlines_ratio):].std()
        loss_report = "{0}eps MA loss: {1:.2f}\n{0}eps STD: {2:.3f}".format(score_window, loss_mean, loss_std)

        fig = plt.figure(figsize=(20,10))
        gs = GridSpec(2, 2, hspace=.5, wspace=.2, top=dtop-0.08)
        ax1 = fig.add_subplot(gs[:,0])
        ax2 = fig.add_subplot(gs[0,1])
        gs2 = GridSpec(1,1, bottom=dtop-0.01, top=dtop)
        dummyax = fig.add_subplot(gs2[0,0])

        # Plot unfiltered scores
        ax1.plot(score_x, self.slines)
        # Plot 200MA line
        ax1.plot(score_x, self._moving_avg(self.slines, score_window*2),
                color=ma200_color, lw=3, label="{}eps MA".format(score_window*2))
         # Plot 100MA line
        ax1.plot(score_x, self._moving_avg(self.slines, score_window),
                color=ma100_color, lw=2, label="{}eps MA".format(score_window))
        ax1.set_title("Scores")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score")
        ax1.set_facecolor((bg_color, bg_color, bg_color))
        ax1.grid()
        ax1.legend(loc="upper left", markerscale=2.5, fontsize=15)
        ax1.axvspan(score_x[-score_window], score_x[-1], color=(0.1,0.4,0.1), alpha=0.25)
        ax1.annotate(score_report, xy=(0,0), xycoords="figure points", xytext=(0.925,0.05),
                    textcoords="axes fraction", horizontalalignment="right",
                    size=20, color='white', bbox = annotate_props)

        # Plot unfiltered network loss data
        ax2.plot(loss_x, self.nlines)
        # Plot 200MA line
        ax2.plot(loss_x, self._moving_avg(self.nlines, score_window*2*nlines_ratio),
                color=ma200_color, lw=3, label="{}eps MA".format(score_window*2))
        # Plot 100MA line
        ax2.plot(loss_x, self._moving_avg(self.nlines, score_window*nlines_ratio),
                color=ma100_color, lw=2, label="{}eps MA".format(score_window))
        ax2.set_xticks(np.linspace(0, len(self.nlines), xcount))
        ax2.set_yticks(n_yticks)
        ax2.set_xticklabels(xticks)
        ax2.set_ylabel("Loss", labelpad=10)
        ax2.set_title("Network Loss")
        ax2.set_facecolor((bg_color, bg_color, bg_color))
        ax2.grid()
        ax2.legend(loc="upper left", markerscale=1.5, fontsize=12)
        ax2.axvspan(loss_x[-int(score_window*nlines_ratio)], loss_x[-1], color=(0.1,0.4,0.1), alpha=0.25)
        ax2.annotate(loss_report, xy=(0,0), xycoords="figure points", xytext=(0.935,0.79),
                    textcoords="axes fraction", horizontalalignment="right",
                    size=14, color='white', bbox = annotate_props)

        dummyax.set_title(self.sess_params, size=13)
        dummyax.axis("off")

        fig.suptitle("{} Training Run".format(self.framework), size=40)

        if save_to_disk:
            save_file = os.path.join(self.save_dir, self.filename+"_graph.png")
            fig.savefig(save_file)
            statement = "Saved graph data to: {}".format(save_file).replace("\\", "/")
            print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        else:
            fig.show()

    def graph(self, logdir=None, save_to_disk=True):
        """
        Preps filepaths and then loads data from on-disk logs. Then graphs them
        for review. If SAVE_TO_DISK is False, then a graph will be popped up but
        not saved. Default is to save to disk and not do a pop-up.
        """

        if logdir != None:
            self.log_dir = logdir
            self.filename = os.path.basename(logdir)
            for f in os.listdir(self.log_dir):
                f = os.path.join(self.log_dir,f)
                if f.endswith("_LOG.txt"):
                    self.paramfile = f
                if f.endswith("_networkloss.txt"):
                    self.netlossfile = f
                if f.endswith("_scores.txt"):
                    self.scoresfile = f
        self.load_logs()
        self.plot_logs(save_to_disk)

    def _generate_logfiles(self, multi_agent):
        """
        Creates empty files for later writing. Creating the empty files isn't
        strictly necessary at this point, but it feels neater.
        """

        self.scoresfile = self.logs_basename + "_scores.txt"
        open(self.scoresfile, 'a').close() # create empty scores file
        self.lossfiles = {}
        statement = []
        for idx, agent in enumerate(multi_agent.agents):
            self.lossfiles[idx] = []
            for name in self._get_agentlog_names(idx, agent):
                file = self.logs_basename + name
                self.lossfiles[idx].append(file)
                open(file, 'a').close() #create empty loss log
                statement.append(file)
        return statement

    def _init_logs(self, params, multi_agent):
        """
        Outputs an initial log of all parameters provided as a list. Initialize
        blank logs for scores, and agent losses.
        """

        # Create the log files. Params is filled on creation, the others are
        # initialized blank and filled as training proceeds.
        leader = "..."
        params_file = self.logs_basename + "_LOG.txt"
        log_statement = ["Logfiles saved to: {}/".format(self.log_dir)]
        log_statement.append(leader + os.path.basename(params_file))
        with open(params_file, 'w') as f:
            for line in params:
                f.write(line + '\n')
        files = self._generate_logfiles(multi_agent)
        # log_statement.append(leader + os.path.basename(file))
        for file in files:
            log_statement.append(leader + os.path.basename(file))
        print_bracketing(log_statement, center=False)

    def _get_agentlog_names(self, idx, agent):
        """
        Creates a unique filename for each agent.
        Param AGENT is currently not used, but is left in for future, more
        robust file naming by pulling network names from the agent instead of
        hardcoded by hand as in the current implmementation.
        """
        idx = idx + 1
        actorlossfile = "_agent{}_actorloss.txt".format(idx)
        criticlossfile = "_agent{}_criticloss.txt".format(idx)
        return (actorlossfile, criticlossfile)

    def _collect_params(self, args, agent):
        """
        Creates a list of all the Params used to run this training instance,
        prints this list to the command line if QUIET is not flagged, and stores
        it for later saving to the params log in the /logs/ directory.
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

    def _runtime(self):
        """
        Return the time since the previous episode, as well as total time for
        the training session.
        """

        current_time = time.time()
        eps_time = self._format_time(current_time, self.prev_timestamp)
        total_time = self._format_time(current_time, self.start_time)
        self.prev_timestamp = current_time
        return eps_time, total_time

    def _format_time(self, current, previous):
        """
        Formats time difference into Hours, Minutes, Seconds.
        """

        m, s = divmod(current - previous, 60)
        h, m = divmod(m, 60)
        return "{}h{}m{}s".format(int(h), int(m), int(s))

    def _update_score(self):
        """
        Calculates the average reward for the previous episode, prints to the
        cmdline, and then saves to the logfile.
        """

        #Rewards are summed over each episode for each agent, during
        score = self.rewards.max()
        # print("{}Return: {}".format("."*10, score))
        if not self.eval:
            self._write_scores(score)
        return score

    def _write_losses(self, multi_agent):
        """
        Writes actor/critic loss data to file.
        """

        for idx, agent in enumerate(multi_agent.agents):
            for filename in self.lossfiles[idx]:
                if "actorloss" in filename:
                    loss = agent.actor_loss
                elif "criticloss" in filename:
                    loss = agent.critic_loss
                else:
                    raise ValueError
                with open(filename, 'a') as f:
                    f.write(str(loss) + '\n')

    def _write_scores(self, score):
        """
        Writes score data to file.
        """

        with open(self.scoresfile, 'a') as f:
            f.write(str(score) + '\n')

    def _init_rewards(self):
        """
        Resets the REWARDS matrix to zero for starting an episode.
        """

        self.rewards = np.zeros(self.agent_count)



def gather_args():
    """
    Generate arguments passed from the command line.
    """
    parser = ArgumentParser(description="Continuous control environment for Udacity DeepRL course.",
            usage="")

    parser.add_argument("-alr", "--actor_learn_rate",
            help="Actor Learning Rate.",
            type=float,
            default=1e-3)
    parser.add_argument("-clr", "--critic_learn_rate",
            help="Critic Learning Rate.",
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
            default=350)
    parser.add_argument("-cpu", "--cpu",
            help="Run training on the CPU instead of the default (GPU).",
            action="store_true")
    parser.add_argument("-e", "--e",
            help="Noisey exploration rate.",
            type=float,
            default=0.2)
    parser.add_argument("-vmin", "--vmin",
            help="Min value of reward projection.",
            type=float,
            default=-0.015)
    parser.add_argument("-vmax", "--vmax",
            help="Max value of reward projection.",
            type=float,
            default=0.15)
    parser.add_argument("-atoms", "--num_atoms",
            help="Number of atoms to project categorically.",
            type=int,
            default=51)
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
            default=25)
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
    # To avoid redundant code checks elsewhere, EVAL should be set to True if
    # FORCE_EVAL is flagged
    if args.force_eval:
        args.eval = True
    # Ensure that ROLLOUT is 1 or greater
    assert args.rollout >= 1, "ROLLOUT must be greater than or equal to 1."

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



def _get_filepath(files):
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
        print_bracketing('Input "{}" is INVALID...'.format(save_file))
        return _get_filepath(files)
