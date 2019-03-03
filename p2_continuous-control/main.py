import time
import numpy as np
import torch
import progressbar
from unityagents import UnityEnvironment
import numpy as np
from logger import Logger

# from agent import Agent
# from utils import Environment, load_checkpoint, print_verbose_info, report_results, print_status, save_checkpoint
# from get_args import get_args


def main():
    """
    D4PG Agent for Udacity's Continuous Control project found at:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control
    """
    args = get_args()

    # logger = Logger()

    env = Environment(args)

    agent = D4PG_Agent(env.state_size,
                       env.action_size,
                       env.agent_count,
                       args.rollout)

    # Run through the environment until the replay buffer has collected a
    # minimum number of trajectories for training
    agent.initialize_memory(args.pretrain, env)

    ### NO NEED FOR MULTIPLE CODED AGENTS, AS ENVIRONMENT PROVIDES 20 ACTORS
    # while len(agent.replay_memory) < config.batch_size:
    #     agent.launch_k_actors
    #     agent.store_experiences

    env.reset()

    for episode in range(args.num_episodes):
        states = env.states
        for t in range(args.max_time):
            actions = agent.act(states)
            rewards, next_states, dones = env.step(actions)

            agent.save_experience(states, actions, rewards, next_states, dones)

            agent.learn()

            states = next_states

    return




if __name__ == "__main__":
    main()
