import numpy as np

from logger import Logger
from agent import D4PG_Agent
from environment import Environment
from meta import Meta
from utils import Saver


def main():
    """
    Originall written for Udacity's Continuous Control project:
    https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control

    This environment utilizes 20 actors built into the environment for parallel
    training. This specific code therefore has no implementation of distributed
    K-Actors training, but it would be straightforward to roll it into this
    training loop as needed.
    """

    meta = Meta()

    env = Environment(meta.args)

    agent = D4PG_Agent(env.state_size,
                       env.action_size,
                       env.agent_count,
                       device = meta.args.device)
                       # a_lr = meta.args.actor_learn_rate,
                       # c_lr = meta.args.critic_learn_rate,
                       # batch_size = meta.args.batch_size,
                       # buffer_size = meta.args.buffer_size,
                       # C = meta.args.C,
                       # gamma = meta.args.gamma,
                       # rollout = meta.args.rollout)

    if meta.load_file: meta.load_agent(agent)

    if meta.args.eval:
        eval(agent, meta.args, env)
    else:
        meta.collect_params(agent)
        train(agent, meta.args, env)

    return True



def train(agent, args, env):
    """
    Train the agent.
    """

    saver = Saver(agent, args)

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

        # Gather experience until done or max_steps is reached
        for t in range(args.max_steps):
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

def eval(agent, args, env):
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

        # Gather experience until done or max_steps is reached
        for t in range(args.max_steps):
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
