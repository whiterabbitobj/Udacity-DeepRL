# import time
import numpy as np
# import torch

# from unityagents import UnityEnvironment
from logger import Logger
# import os
from agent import D4PG_Agent
from environment import Environment
from utils import Saver
# from get_args import get_args
from meta import Meta

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

    meta = Meta()

    args = meta.args

    env = Environment(args)

    agent = D4PG_Agent(env.state_size,
                       env.action_size,
                       env.agent_count,
                       a_lr = args.actor_learn_rate,
                       c_lr = args.critic_learn_rate,
                       batch_size = args.batch_size,
                       buffer_size = args.buffer_size,
                       C = args.C,
                       device = args.device,
                       gamma = args.gamma,
                       rollout = args.rollout)

    saver = Saver(agent, args)
    if meta.load_file: meta.load_agent(agent)

    if args.eval:
        eval(args, env, agent)
    else:
        train(agent, args, env, saver)

    return True



def train(agent, args, env, saver):
    """
    Train the agent.
    """

    logger = Logger(agent, args, env)

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
        # PRINT DEBUGGING INFO AFTER EACH EPISODE
        print("A LOSS: ", agent.actor_loss)
        print("C LOSS: ", agent.critic_loss)

    env.close()
    saver.save_final(agent)
    #logger.report(args.save_dir)
    #logger.print_results()
    return True

def eval(args, env, agent):
    """
    Evaluate the performance of an agent using a saved weights file.
    """

    #logger = Logger(agent, args, env)

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
            states = next_states
            #logger.rewards += rewards
            if np.any(dones):
                break
        agent.new_episode()
        #logger.step(episode)

    env.close()
    return True

if __name__ == "__main__":
    main()
