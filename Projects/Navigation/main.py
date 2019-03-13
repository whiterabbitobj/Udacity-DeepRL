import numpy as np

from agent import DQN_Agent
from environment import Environment
from data_handling import Logger, Saver, gather_args

def main():
    """
    Primary code for training or testing an agent using one of several
    optional network types. Deep Q-Network, Double DQN, etcself.

    This is a project designed for Udacity's Deep Reinforcement Learning Nanodegree
    and uses a special version of Unity's Banana learning environment. This
    environment should be available via the github repository at:
    https://github.com/whiterabbitobj/udacity-deep-reinforcement-learning/tree/master/p1_navigation
    """

    """
    Sets up a few global variables to condense code:
        args - arguments from command line, including defaults
        sep - separator used for print statements
        start_time - start time of program initialization
        device - which device to run on (usually GPU)    """

    args = gather_args()

    env = Environment(args)

    agent = DQN_Agent(env.state_size,
                      env.action_size,
                      args)

    saver = Saver(agent.framework, agent, args.save_dir, args.load_file)

    if args.eval:
        eval(agent, args, env)
    else:
        train(agent, args, env, saver)

    return True



def train(agent, args, env, saver):
    """
    Train the agent.
    """

    logger = Logger(agent, args, saver.save_dir, log_every=50, print_every=5)

    # Pre-fill the Replay Buffer
    agent.initialize_memory(args.pretrain, env)

    #Begin training loop
    for episode in range(1, args.num_episodes+1):
        # Begin each episode with a clean environment
        done = False
        env.reset()
        # Get initial state
        state = env.state
        # Gather experience until done or max_steps is reached
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if done:
                next_state = None
                # print("terminal state reached!")
            # if reward:
            #     print("{} ".format(reward), end="")
            agent.step(state, action, reward, next_state)
            state = next_state

            logger.log(reward, agent)


        saver.save_checkpoint(agent, args.save_every)
        agent.new_episode()
        logger.step(episode, agent.epsilon)

    env.close()
    saver.save_final(agent)
    logger.graph()
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
