[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"



# **Project 3: Collaboration and Competition**

#### Deep Reinforcement Learning Nanodegree culminates in:
### **[Multi-agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275)**
###### _combined with_
### **[Distributed Distributional Deep Deterministic Policy Gradient (D4PG)](https://arxiv.org/pdf/1804.08617.pdf)**
###### _to create_
## <u>_**MAD4PG**_</u>
---

This project uses a state-of-the-art multi-agent interaction algorithm, MADDPG, pioneered by OpenAI, to solve an environment in which two agents compete to play Tennis. Scoring is based on back and forth volleying and not on beating the opponent, so this environment is really more focused on _collaboration_ than competition.

MADDPG presents a rigorous theoretical background for how multiple DDPG agents can interact and learn in the same environment, presenting solutions to non-stationary environments and estimating unknown adversary policies, as well as an approach to more robust and stable training utilizing _ensemble policies_.

Ensemble policies are not implemented here. Nor is policy estimation required as the environment provides access to all agents' observations. Estimating unknown policies would be an area of important future implementation for more complex or real-world oriented tasks.

This implementation combines _**MADDPG**_ with _**D4PG**_ for more robust training than DDPG provides on it's own. This agent will be henceforth always referred to as MAD4PG.

D4PG uses distributed training, distributional value estimation, n-step bootstrapping, and an Actor-Critic architecture to provide fast, stable training in a continuous action space for which traditional Q-value estimation would fail. The D4PG paper also discusses the implementation of a Prioritized Experience Replay buffer, but it was not necessary to implement for this project to still achieve very fast, stable results.

### **Instructions**

This agent can be trained or evaluated from the same python file **`main.py`**.

#### Running the Agent
The default values for the agent's params are set to great training values, and running this agent is as simple as:  
**`python main.py`**  
If you wish to see the trained agent in action, you can run:  
**`python main.py -eval`**  
Please see the REPORT file notebook for additional information!

> Parameters associated with the agent networks
>
> **`-layers/--layer_sizes`** - Entry should be two integers specifying the size of the network hidden layers. The agent currently accept 2 and only 2 layers, which was sufficient for the Udacity environments but likely will need flexibility for more complex tasks.  
> **`-alr/--actor_learn_rate`** - adjust the learning rate of the Actor network.  
> **`-clr/--critic_learn_rate`** - adjust the learning rate of the Critic network.  
> **`-C/--C`** - Number of timesteps between updating the network if training with HARD updates.  
> **`-t/--tau`** - Rate of soft transfer of network weights when training with SOFT updates.  

> Please review the section of the REPORT file which deals with Epsilon for further details on the implementation of Epsilon annealing motivated by agent performance.  
>
> **`-e/--e`** - Maximum value of the scalar value multiplied against the gaussian noise used to encourage exploration.  
> **`-em/--e_min`** - Minimum value to which `e` will anneal over time.  
> **`-amax/--anneal_max`** - The cumulative episode reward at which `e` annealing will reach its minimum.

> Please see the section of the REPORT detailing Distributional Value Estimation for further details about these parameters.
>
> **`-vmin/--vmin -vmax/--vmax`** - Lower/Upper bounds of the value distribution calculated by the Critic network.  
> **`-atoms/--num_atoms`** - How many discrete probabilities for the Critic network to calculate, to be combined with vmin/vmax to determine state-action values.  

> General useful parameters  
> **`-rollout/--rollout`** - The length of n-step bootstrapping to calculate for stored memories.  
> **`-pre/--pretrain`** - How many experiences utilizing random actions should be collected before the Agent begins to learn.  
> **`-bs/--batch_size`** - size of ReplayBuffer samples.  
> **`--eval`** - Used to watch a trained agent. If EVAL not flagged, TRAINING is the default mode for this code. EVAL will prompt the user to select a savefile, unless --latest is flagged, which will naively choose the most recently created savefile in the save_dir.  
> **`--quit_threshold`** - If the average score reaches this value, the training will early-abort. Should obviously be above or well-above the environment goal/solve score.


## **The Environment**


This project utilizes a special, Udacity created version of the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

(Click below for video)  
[!["Trained Agent"](http://img.youtube.com/vi/d-r1OjuTpaI/0.jpg)](https://youtu.be/d-r1OjuTpaI "Reacher")

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, the environment contains 20 identical agents, simulating a K-Actors implementation and negating the need to launch multiple training instances via multi-threading (thanks Udacity!). Thus, each timestep provides 20 experiences to the ReplayBuffer. Learning is not performed 20 times per timestep, it is only performed a single time with a batch from the buffer, as per K-Actor methodology.

This is useful for any algorithm (such as [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), or [D4PG](https://arxiv.org/pdf/1804.08617.pdf)) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

The environment is considered solved when the average score over the previous 100 episodes is at least +30 when averaging the returns across all agents in the environment.

This D4PG agent reliably scores well above this minimum goal. When evaluated with noise no longer added to the Actor's action choices, the Agent can achieve near-perfect performance, which even looks human-level smooth. Take a look at the YouTube preview above!

### **Dependencies**

1. The environment for this project is included in the repository as both a Window environment and Linux environment.

    The environment can also be downloaded from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. The environment folder should live in the base directory with the python files. This github repository can be used as-is if cloned.  

3. Use pip to install `requirements.txt`

4. Use Conda to install the `environment.yml` file.

5. In some environments, such as Linux, some of the packages may throw errors. It should be safe to safely remove these.

6. There are no unusual packages utilized and a clean environment with Python 3.6+, Pytorch, Matplotlib, and Numpy installed should suffice.


### _(Optional) Challenge: Crawler Environment_

Udacity has provided a more difficult **Crawler** environment.

![Crawler][image2]

This agent has not yet been tested against this more complex environment... but it's next on the TO-DO list immediately after finishing the class!
