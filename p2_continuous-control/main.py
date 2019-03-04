import time
import numpy as np
import torch
import progressbar
from unityagents import UnityEnvironment
import numpy as np
from logger import Logger

from agent import D4PG_Agent
from environment import Environment
# from utils import Environment, load_checkpoint, print_verbose_info, report_results, print_status, save_checkpoint
# from get_args import get_args


def main():
    """
    Algorithm implementation based on the original paper/research by
    Barth-Maron, Hoffman, et al: https://arxiv.org/abs/1804.08617

    D4PG Agent for Udacity's Continuous Control project:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

    This environment utilizes 20 actors built into the environment for parallel
    training. This specific code therefore has no implementation of K-Actors
    training that is discussed in the original D4PG paper.
    """
    args = get_args()

    env = Environment(args)

    logger = Logger(env)

    agent = D4PG_Agent(env.state_size,
                       env.action_size,
                       env.agent_count,
                       args.rollout)

    # Run through the environment until the replay buffer has collected a
    # minimum number of trajectories for training
    agent.initialize_memory(args.pretrain, env)

    # Ensure that the environment is in it's starting state before training
    env.reset()

    #Begin training loop
    for episode in range(1, args.num_episodes+1):
        states = env.states

        # Gather experience for a maximum amount of steps, or until Done,
        # whichever comes first
        for t in range(args.max_time):
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states)
            states = next_states

            logger.rewards += rewards
            if np.any(dones):
                break

        logger.log_score()


    env.close()
    logger.report()
    #logger.print_results()
    return




if __name__ == "__main__":
    main()
