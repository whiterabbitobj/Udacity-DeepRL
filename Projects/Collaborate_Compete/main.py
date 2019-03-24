# -*- coding: utf-8 -*-
import numpy as np

from agent import MAD4PG_Net
from environment import Environment
from data_handling import Logger, Saver, gather_args

def main():
    """
    Originall written for Udacity's Collaborate/Compete project:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

    This project utilizes multi-agent reinforcement learning techniques to
    teach two agents to play table tennis against each other in an environment
    limited to two axis (X, Y).

    The implementation for class uses a variant of OpenAI's MADDPG:
    https://arxiv.org/abs/1706.02275
    but utilizes D4PG as a base algorithm:
    https://arxiv.org/pdf/1804.08617.pdf
    in what I will call MAD4PG.

    The improvements over DDPG include multi-step rollout, and distributional
    reward prediction, for faster and more stable learning.

    A more robust, feature complete and multi-application version of this
    implementation may follow for non-Udacity environments, in a separate
    repository. For instance, more flexible multi-agent handling, instead of the
    hard-coded two agents for this specific application.
    """

    args = gather_args()

    env = Environment(args)

    multi_agent = MAD4PG_Net(env, args)

    saver = Saver(multi_agent, args)

    if args.train:
        train(multi_agent, args, env, saver)
    else:
        eval(multi_agent, args, env)

    return True



def train(multi_agent, args, env, saver):
    """
    Train the agent.
    """

    logger = Logger(multi_agent, args, saver.save_dir)

    # Pre-fill the Replay Buffer
    multi_agent.initialize_memory(args.pretrain, env)

    #Begin training loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        observations = env.states
        # Gather experience until done or max_steps is reached
        while True:
            actions = multi_agent.act(observations)
            next_observations, rewards, dones = env.step(actions)
            # print(len(rewards), len(dones))
            # multi_agent.store((ob))
            multi_agent.step(observations, actions, rewards, next_observations, dones)
            observations = next_observations
            # print(rewards)
            logger.log(rewards, multi_agent)
            if np.any(dones):
                break

        saver.save_checkpoint(multi_agent, args.save_every)
        multi_agent.new_episode()
        logger.step(episode, multi_agent)

    env.close()
    saver.save_final(multi_agent)
    logger.graph()
    return True

def eval(multi_agent, args, env):
    """
    Evaluate the performance of an agent using a saved weights file.
    """

    logger = Logger(multi_agent, args)

    #Begin evaluation loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        observations = env.states
        # Gather experience until done or max_steps is reached
        while True:
            actions = multi_agent.act(observations)
            next_observations, rewards, dones = env.step(actions)
            observations = next_observations

            logger.log(rewards)
            if np.any(dones):
                break

        multi_agent.new_episode()
        logger.step()

    env.close()
    return True

if __name__ == "__main__":
    main()
