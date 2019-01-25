import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train or Test a Deep RL agent in Udacity's Banana Environment",
            usage="EXAMPLE COMMAND:\npython banana_agent.py --train --batch_size 64 -lr 5e-4")

    parser.add_argument("-m", "--mode",
            help="In which mode should the Agent run? (train, demo)")
    parser.add_argument("-a", "--agent_type",
            help="Which type of Agent to use. (DQN, DDQN)")
    parser.add_argument("-bs", "--batchsize",
            help="Size of each batch between learning updates",
            type=float,
            default=64)
    parser.add_argument("-buffer", "--buffersize",
            help="How many past timesteps to keep in memory.",
            type=int,
            default=int(1e5))
    parser.add_argument("-ec", "--episode_count",
            help="How many episodes to train?",
            type=int,
            default=1500)
    parser.add_argument("-e", "--epsilon",
            help="Starting value of Epsilon.",
            type=float,
            default=1.0)
    parser.add_argument("-ed", "--epsilon_decay",
            help="Epsilon decay value.",
            type=float,
            default=0.999)
    parser.add_argument("-em", "--epsilon_min",
            help="Minimum value for epsilon.",
            type=float,
            default=0.1)
    parser.add_argument("-gamma",
            help="Gamma (Discount rate).",
            type=float,
            default=0.99)
    parser.add_argument("-lr", "--learn_rate",
            help="Alpha (Learning Rate).",
            type=float,
            default=5e-4)
    parser.add_argument("-tau",
            help="Tau",
            type=float,
            default=1e-3)
    parser.add_argument("-u", "--update_every",
            help="Timesteps between acting on new environment states.",
            type=int,
            default=4)

    return parser.parse_args()
