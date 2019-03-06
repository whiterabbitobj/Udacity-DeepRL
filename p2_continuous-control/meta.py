import Logger
import Saver

from utils import print_bracketing


class Meta():
    def __init__(self):
        self.args = self.get_args()
        
        process_args()

        self.saver =


        pass

    # def


    def process_args(self):
        self.args.train = not args.eval
        assert args.pretrain >= args.batch_size, "PRETRAIN less than BATCHSIZE."

        if args.eval and args.num_episodes > 10:
            print("In eval mode, num_episodes is set to not more than 10.")
            args.num_episodes = 10

        if not args.quiet:
            arg_print = ''
            for arg in vars(args):
                if arg == "quiet": continue
                arg_print += " "*12 + "{}: {}\n".format(arg.upper(), getattr(args, arg))
            print_bracketing(arg_print[:-1])


    def get_args():
        parser = argparse.ArgumentParser(description="Continuous control environment for Udacity DeepRL course.",
                usage="")

        parser.add_argument("-alr", "--actor_learn_rate",
                help="Alpha (Learning Rate).",
                type=float,
                default=1e-4)
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
                default=100000)
        parser.add_argument("-C", "--C",
                help="How many timesteps between hard network updates.",
                type=int,
                default=4000)
        parser.add_argument("-eval", "--eval",
                help="Run in evalutation mode. Otherwise, will utilize training mode.",
                action="store_true")
        parser.add_argument("-gamma",
                help="Gamma (Discount rate).",
                type=float,
                default=0.99)
        parser.add_argument("-max", "--max_time",
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
                default=1500)
        parser.add_argument("-pre", "--pretrain",
                help="How many trajectories to randomly sample into the ReplayBuffer\
                      before training begins.",
                type=int,
                default=1000)
        parser.add_argument("--quiet",
                help="Print less while running the agent.",
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
                default="")
        parser.add_argument("-savedir", "--save_dir",
                help="Directory to find saved agent weights.",
                type=str,
                default="saves")
        return parser.parse_args()
