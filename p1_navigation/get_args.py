import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train or Test a Deep RL agent in Udacity's Banana Environment",
            usage="EXAMPLE COMMAND:\n\
                   python banana_deeprl.py --train --pixels --verbose --framework DDQN -ed .999 -em .05 -lr .00025 -C 2000 -u 3 -buffer 10000 -num 2000\
                   python banana_deeprl.py --train --verbose -no_per -ed .991 -em 0.01 -lr 0.0005 -C 2000 -buffer 30000 --pixels\
                   ")
    parser.add_argument("-f", "--framework",
            help="Which type of Agent to use. (DQN, D2DQN (double dqn), DDQN (dueling dqn))",
            type=str,
            default="DDQN")
    parser.add_argument("-no_per", "--no_prioritized_replay",
            help="Use standard Replay Buffer instead of Prioritized Experience Replay.",
            action="store_true")
    parser.add_argument("-a", "--alpha",
            help="Alpha parameter of the Prioritized Experience Replay.",
            type=float,
            default=0.6)
    parser.add_argument("-b", "--beta",
            help="Beta parameter of the Prioritized Experience Replay.",
            type=float,
            default=0.4)
    parser.add_argument("-bs", "--batchsize",
            help="Size of each batch between learning updates",
            type=int,
            default=64)
    parser.add_argument("-buffer", "--buffersize",
            help="How many past timesteps to keep in memory.",
            type=int,
            default=25000)
    parser.add_argument("-C",
            help="How many timesteps between updating Q' to match Q",
            type=int,
            default=600)
    parser.add_argument("--continue",
            help="Continue training from a loaded file (can use in conjunction with --latest).",
            action="store_true")
    parser.add_argument("-drop", "--dropout",
            help="Dropout rate for deep network.",
            type=float,
            default=0.0)
    parser.add_argument("-e", "--epsilon",
            help="Starting value of Epsilon.",
            type=float,
            default=1.0)
    parser.add_argument("-ed", "--epsilon_decay",
            help="Epsilon decay value.",
            type=float,
            default=0.99)
    parser.add_argument("-em", "--epsilon_min",
            help="Minimum value for epsilon.",
            type=float,
            default=0.01)
    parser.add_argument("-fs", "--framestack",
            help="How many recent frames to stack for temporal replay.",
            type=int,
            default=4)
    parser.add_argument("-skip", "--frameskip",
            help="How many frames to skip in between new actions/frame stacking.",
            type=int,
            default=4)
    parser.add_argument("-gamma",
            help="Gamma (Discount rate).",
            type=float,
            default=0.99)
    parser.add_argument("--latest",
            help="Use this flag to automatically use the latest save file to \
                  run in DEMO mode (instead of choosing from a prompt).",
            action="store_true")
    parser.add_argument("-lr", "--learn_rate",
            help="Alpha (Learning Rate).",
            type=float,
            default=0.001)
    parser.add_argument("-m", "--momentum",
            help="Momentum for use in specific optimizers like SGD",
            type=float,
            default=0.95)
    parser.add_argument("--nographics",
            help="Run Unity environment without graphics displayed.",
            action="store_true")
    parser.add_argument("-num", "--num_episodes",
            help="How many episodes to train?",
            type=int,
            default=1500)
    parser.add_argument("-o", "--optimizer",
            help="Choose an optimizer for the network. (RMSprop/Adam/SGD)",
            type=str,
            default="Adam")
    parser.add_argument("--pixels",
            help="Train the network using visual data instead of states from the engine.",
            action="store_true")
    parser.add_argument("--print_count",
            help="How many times to print status updates during training. \
                  Bounded between 2<->100.",
            type=int,
            default=15)
    # parser.add_argument("-tau",
    #         help="Tau",
    #         type=float,
    #         default=1e-3)
    parser.add_argument("-t", "--train",
            help="Run in training mode.",
            action="store_true")
    parser.add_argument("-u", "--update_every",
            help="Timesteps between updating the network parameters.",
            type=int,
            default=4)
    parser.add_argument("-v", "--verbose",
            help="Print additional information while running the agent.",
            action="store_true")

    return parser.parse_args()
