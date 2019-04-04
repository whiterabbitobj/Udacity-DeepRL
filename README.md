[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
<div align="justify">

# **Deep Reinforcement Learning Nanodegree**

![Trained Agents][image1]

---

## **Summary**

This repository contains the classwork required to pass and demonstrate knowledge from the Udacity's _[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)_ program.

Deep Reinforcement Learning is at the forefront of Artificial Intelligence research and progress and I personally believe will be the key instrumental field towards evetual _artificial general intelligence_, an important milestone for humanity about which I am passionate and would like to contribute towards in some modest way.

The projects contained here are both simple and not simple. These specific environments contain well established methods for completing, and while the ability to set the stage, step back, and watch a computer learn on its own is compelling in the extreme, I look forward to moving on from here to more real-world applicable tasks.

  #### Goals for the future:
  * Each project has a detailed _REPORT_ with desires for future learning based on each specific framework and task
  * There is a huge number of agent frameworks and algorithms publicly available for study and research. I would especially like to implement:
    * [TRPO](https://arxiv.org/abs/1502.05477)
    * [PPO](https://arxiv.org/abs/1707.06347)  
    and, in my opinion, extremely promising, the concept of
    * [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) _**Capsule Networks**_  

  When these agents are implemented, they will be in another repository.

---

## **Labs / Projects**

The labs and projects can be found below.  All of the projects use simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

Navigation _(Banana Collector)_, Continuous Control _(Reacher)_, and Collaborate & Compete _(Tennis)_ were all reviewed and passed the Udacity grading process on the first submission, with complimentary remarks on the performance and implementation.

* [Navigation](https://github.com/whiterabbitobj/Udacity-DeepRL/tree/master/Projects/Navigation): In the first project, train an agent to collect yellow bananas while avoiding blue bananas.  
* [Continuous Control](https://github.com/whiterabbitobj/Udacity-DeepRL/tree/master/Projects/Continuous_Control): In the second project, train robotic arms to reach target locations.  
* [Collaboration and Competition](https://github.com/whiterabbitobj/Udacity-DeepRL/tree/master/Projects/Collaborate_Compete): In the third project, train a pair of agents to play tennis.
  ##### Additional Challenges (to be implemented soon):
  * _Crawler_: Building on the principles learned in the _reacher_ environment, this environment teaches a quadrupedal agent to move towards a goal point.
  * _Soccer_: Building on the principles learned in the _tennis_ environment, this environment teaches four agents to compete in a soccer match in a much more complex example of ally/adversary behavior.

### Resources

* [Cheatsheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet): This provided Reinforcement Learning overview cheatsheet was invaluable in getting started early in the class work.
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf): _**The**_ book to read, authored by two of the pioneers of the field and as up-to-date as any published material can be in a field that moves this quickly.

  ##### Papers referenced in this repo:
  * [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (DQN)
  * [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461) (DDQN)
  * [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (DDPG)
  * [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617) (D4PG)
  * [Multi-agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) (MADDPG)
  * [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)

## **Dependencies**

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name <NAME> python=3.6
	source activate <NAME>
	```
	- __Windows__:
	```bash
	conda create --name <NAME> python=3.6
	activate <NAME>
	```

2. Use pip to install `requirements.txt`

3. Use Conda to install the `environment.yml` file.

4. In some environments, such as Linux, some of the packages may throw errors. It should be safe to safely remove these.

5. There are no unusual packages utilized and a clean environment with Python 3.6+, Pytorch, Matplotlib, and Numpy installed should suffice.


## Want to learn more?

<p align="center"><a href="http://www.udacity.com">Udacity</a> has been an incredible resource for learning this exciting field in a clear, organized manner, and I highly recommend their classes to anyone!  

This repo contains the results and summary of the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
