import time
import numpy as np
import torch
import progressbar
from unityagents import UnityEnvironment
import numpy as np


# from agent import Agent
# from utils import Environment, load_checkpoint, print_verbose_info, report_results, print_status, save_checkpoint
# from get_args import get_args


def main():
    """
    D4PG Agent for Udacity's Continuous Control project found at:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control
    """
    args = get_args()

    env = Environment(args)

    agent = D4PG_Agent(env.state_size,
                       env.action_size,
                       env.agent_count,
                       args.rollout)
    # agent.launch_k_actors

    agent.initialize_memory(args.batchsize, env)

    ### NO NEED FOR MULTIPLE CODED AGENTS, AS ENVIRONMENT PROVIDES 20 ACTORS
    # while len(agent.replay_memory) < config.batch_size:
    #     agent.launch_k_actors
    #     agent.store_experiences

    env.reset()
    
    for episode in range(args.num_episodes):

        for t in range(args.max_time):
            trajectory = agent.collect_trajectory(env, args.rollout)
            agent.memory.add(trajectory)
            agent.learn()


            batch = agent.memory.sample(args.batchsize)
            action = agent.act()
            rewards, done = env.step(action)


            for n in range(args.rollout_num):

            states, actions, rewards, next_states, dones = batch





    return




if __name__ == "__main__":
    main()
