[//]: # (Image References)
<div align="justify">

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
# **Project 1: Navigation**

## *Introduction to Deep Reinforcement Learning*
## **DQN/Double-DQN Implementation**

This project uses vanilla Deep Q-Learning (DQN), or (by default) a slightly more advanced Double Deep Q-Learning Agent (DDQN), to train a Reinforcement Learning Agent. Future implementations may include Dueling DQN, Prioritized Experience Replay, or a _Rainbow_ implementation using features from all of these networks.


### **Instructions**

This agent can be trained or evaluated from the same python file **main.py.**

#### Running the Agent
The default values for the agent's params are set to great training values, and running this agent is as simple as:  
**`python main.py`**  
If you wish to see the trained agent in action, you can run:  
**`python main.py -eval`**  
Please see the REPORT file notebook for additional information!

Use _`python main.py --help`_ for full details of available parameters, but these are the most important for training in the Banana environment:

**`-lr/--learn_rate`** - adjust the learning rate of the Q network.  
**`-bs/--batch_size`** - size of ReplayBuffer samples.  
**`--eval`** - Used to watch a trained agent. If EVAL not flagged, TRAINING is the default mode for this code. EVAL will prompt the user to select a savefile, unless --latest is flagged, which will naively choose the most recently created savefile in the save_dir.  
**`-e/--epsilon`** - Rate at which a random action should be chosen instead of the agent returning an action.  
**`-ed/--epsilon_decay`** - Rate at which epsilon reduces to encourage greedy exploitation.  
**`-pre/--pretrain`** - How many experiences utilizing random actions should be collected before the Agent begins to learn.  
**`-C/--C`** - Number of timesteps between updating the network if training with HARD updates.  
**`-t/--tau`** - Rate of soft transfer of network weights when training with SOFT updates.  


### **The Environment**

This project will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four _**discrete**_ actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and is considered solved when the Agent reaches an average score of +13 over 100 consecutive episodes.

### **Dependencies**

1. The environment for this project is included in the repository as both a Window environment and Linux environment. There is a "Visual" environment that can be used to train the agent from pixel data using a Convolutional Network. This functionality is limited in this code implementation but is present.

    The environment can also be downloaded from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)  


2. The environment folder should live in the base directory with the python files. This github repository can be used as-is if cloned.  

3. Use pip to install `requirements.txt`

4. Use Conda to install the `environment.yml` file.

5. In some environments, such as Linux, some of the packages may throw errors. It should be safe to safely remove these.

6. There are no unusual packages utilized and a clean environment with Python 3.6+, Pytorch, Matplotlib, and Numpy installed should suffice.

### _**Challenge: Learning from Pixels**_

This agent can be trained using pixel-data with a convolutional network using the:
**`--pixels`** flag. There is a special environment required for this training that is also included in the repository, or can be downloaded below:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

This codebase is **extremely inconsistent** with convolutional network training and has not been fully optimized or implemented. While it does run, this optional challenge for Udacity's project was superceded by other projects in more advanced settings.
