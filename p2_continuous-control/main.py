import time
import numpy as np
import torch
import progressbar
from unityagents import UnityEnvironment
import numpy as np
from logger import Logger

from agent import D4PG_Agent
from environment import Environment
from utils import Saver
from get_args import get_args


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

    agent = D4PG_Agent(env.state_size,
                       env.action_size,
                       env.agent_count,
                       a_lr = args.actor_learn_rate,
                       c_lr = args.critic_learn_rate,
                       batch_size = args.batch_size,
                       buffer_size = args.buffer_size,
                       C = args.C,
                       gamma = args.gamma,
                       rollout = args.rollout)

    logger = Logger(env, args.num_episodes)
    saver = Saver(agent)

    # Pre-fill the Replay Buffer
    agent.initialize_memory(args.pretrain, env)

    logger.start_clock()

    #Begin training loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        states = env.states

        # Gather experience until done or max_time is reached
        for t in range(args.max_time):
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states)
            states = next_states

            logger.rewards += rewards
            if np.any(dones):
                break

        saver.save_checkpoint(agent, args.save_every)
        agent.new_episode()
        logger.step(episode)

    env.close()
    saver.save_final(agent)
    #logger.report()
    #logger.print_results()
    return




if __name__ == "__main__":
    main()
