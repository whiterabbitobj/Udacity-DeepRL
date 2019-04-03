import os
import os.path
import matplotlib.pyplot as plt
from PIL import Image
import main
from agent import DQN_Agent
from data_handling import Saver, Logger

def print_args(args):
    print('\n'.join(["{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)]))

def plot_results(imdir):
    images = [os.path.join(imdir, file) for file in os.listdir(imdir) if os.path.splitext(file)[1] == '.png']
    num = len(images)
    for img in images:
        display(Image.open(img))

def print_env_info(state, action, reward):
    print("The agent chooses ACTIONS that look like:\n{}\n".format(action))
    print("The environment returns STATES that look like:\n{}\n".format(state))
    print("The environment returns REWARDS that look like:\n{}".format(reward))

def notebook_eval_agent(args, env, filename, num_eps=2):
    eval_agent = DQN_Agent(env.state_size, env.action_size, args)
    eval_saver = Saver(eval_agent.framework, eval_agent, args.save_dir, filename)
    args.eval = True
    logger = Logger(eval_agent, args)
    for episode in range(3):
        env.reset()
        state = env.state
        for t in range(args.max_steps):
            action = eval_agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            logger.log(reward, eval_agent)
            if done:
                break
        eval_agent.new_episode()
        logger.step(episode)
