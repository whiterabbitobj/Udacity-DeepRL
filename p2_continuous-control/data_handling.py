import sys
import numpy as np
import time
from utils import print_bracketing

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
        self._check_dir(save_dir)
        return save_dir, filename

    def save_checkpoint(self, agent, save_every):
        """
        Saves the current Agent networks to checkpoint files.
        """

        if agent.episode % save_every:
            return
        # checkpoint_dir = os.path.join(self.save_dir, self.filename)
        #self._check_dir(checkpoint_dir)
        save_name = "{}_eps{}_ckpt{}".format(self.filename, agent.episode, self.file_ext)
        full_name = os.path.join(self.save_dir, save_name).replace('\\','/')
        statement = "Saving Agent checkpoint to: {}".format(full_name)
        print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
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
                 agent,
                 args,
                 save_dir = '.',
                 log_every = 10):
        self.max_eps = args.num_episodes
        self.quietmode = args.quiet
        self.log_every = log_every
        self.agent_count = agent.agent_count
        # self.current_log = ''
        # self.full_log = ''
        # self.agent_count = agent.agent_count
        self.save_dir = save_dir
        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.filename = os.path.basename(self.save_dir)
        # self.param_list = self._collect_params(args, agent)

        self._init_rewards()
        self._init_logs(self._collect_params(args, agent))

        statement = "Starting training at: {}".format(time.strftime("%H:%M:%S", time.localtime()))
        print_bracketing(statement)
        self.start_time = self.eps_time =  time.time()
        self.scores = []
        self.losses = []


    def log(self, rewards, agent):
        self.rewards += rewards
        if agent.t_step % self.log_every:
            self._write_losses(agent)
            print("A LOSS: ", agent.actor_loss)
            print("C LOSS: ", agent.critic_loss)
            #self._write_criticloss()
            #self._write_episode_score()

    def _check_dir(self, dir):
        """
        Creates requested directory if it doesn't yet exist.
        """
        if not os.path.isdir(dir):
            os.mkdir(dir)

    def _init_logs(self, params):
        """
        Outputs an initial log of all parameters provided as a list.
        """
        basename = os.path.join(self.log_dir, self.filename)
        paramfile = basename + "_LOG.txt"
        self.alossfile = basename + "_actorloss.txt"
        self.clossfile = basename + "_criticloss.txt"
        self.scoresfile = basename + "_scores.txt"
        with open(paramfile, 'w') as f:
            for line in params:
                f.write(line)
        with open(self.alossfile, 'w') as f:
            pass
        with open(self.clossfile, 'w') as f:
            pass
        with open(self.scoresfile, 'w') as f:
            pass
        log_statement = ["Logfiles saved to: {}".format(self.log_dir)]
        log_statement.append("-{}".format(paramfile))
        log_statement.append("-{}".format(self.alossmfile))
        log_statement.append("-{}".format(self.clossfile))
        log_statement.append("-{}".format(self.scoresfile))
        print_bracketing(log_statement)

    def _write_losses(self, agent):
        with open(self.alossfile, 'a') as f:
            f.write(str(agent.actor_loss) + '\n')
        with open(self.clossfile, 'a') as f:
            f.write(str(agent.critic_loss) + '\n')

    def _write_scores(self, score):
        with open(self.scoresfile, 'a') as f:
            f.write(str(score) + '\n')


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

    def _format_param(self, arg, args):
        """
        Formats into PARAM: VALUE for reporting. Strips leading underscores for
        placeholder params where @properties are used for the real value.
        """
        return "{}: {}\n".format(arg.upper().lstrip("_"), getattr(args, arg))

    def step(self, epsnum):
        epstime, total = self._runtime()
        print("\nEpisode {}/{}... RUNTIME: {}, TOTAL:".format(epsnum, self.max_eps, epstime, total))
        self._update_score()
        self._init_rewards()

    def _runtime(self):
        nowTime = time.time()

        m, s = divmod(nowTime - self.eps_time, 60)
        h, m = divmod(m, 60)
        total = "{}h{}m{}s".format(int(h), int(m), int(s))

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
            default=100)
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
