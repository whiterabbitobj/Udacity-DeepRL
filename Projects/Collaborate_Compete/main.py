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

    multi_agent = MAD4PG_Net(env, args, agent_count=2)

    saver = Saver(multi_agent.framework, multi_agent, args.save_dir, args.load_file)

    if args.eval:
        eval(multi_agent, args, env)
    else:
        train(multi_agent, args, env, saver)

    return True



def train(multi_agent, args, env, saver):
    """
    Train the agent.
    """

    logger = Logger(multi_agent, args, saver.save_dir, log_every=50, print_every=5)

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
            multi_agent.step(observations, actions, rewards, next_observations, dones)
            observations = next_observations

            if np.any(dones):
                break
            # logger.log(rewards, multi_agent)

        saver.save_checkpoint(multi_agent, args.save_every)
        multi_agent.new_episode()
        # logger.step(episode, multi_agent.epsilon)

    env.close()
    saver.save_final(multi_agent)
    # logger.graph()
    return True

def eval(agent, args, env):
    """
    Evaluate the performance of an agent using a saved weights file.
    """

    logger = Logger(agent, args)

    #Begin evaluation loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        env.reset()
        # Get initial state
        state = env.state
        # Gather experience until done or max_steps is reached
        for t in range(args.max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state

            logger.log(reward, agent)
            if done:
                break

        agent.new_episode()
        logger.step(episode)

    env.close()
    return True

if __name__ == "__main__":
    main()