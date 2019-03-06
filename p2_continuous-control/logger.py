import sys
import numpy as np
import time
from utils import print_bracketing

class Logger:
    def __init__(self, env, max_eps):
        self.max_eps = max_eps
        self.current_log = ''
        self.full_log = ''
        self.agent_count = env.agent_count
        self.scores = []
        self._reset_rewards()
        self.t_step = 1

    def add(self, log):
        self.current_log += str(log)

    def start_clock(self):
        t = time.localtime()
        statement = "Starting training at: {}".format(time.strftime("%H:%M:%S", time.localtime()))
        print_bracketing(statement)
        self.start_time = time.time()

    def step(self, eps):
        print("\nEpisode {}/{}... RUNTIME: {}".format(eps, self.max_eps, self._runtime()))
        self._update_score()
        self._reset_rewards()

    def _runtime(self):
        m, s = divmod(time.time() - self.start_time, 60)
        h, m = divmod(m, 60)
        return "{}h{}m{}s".format(int(h), int(m), int(s))

    def _update_score(self):
        score = self.rewards.mean()
        print("{}Return: {}".format("."*10, score))
        self.scores.append(score)

    def _reset_rewards(self):
        self.rewards = np.zeros(self.agent_count)

    def print(self):
        # flushlen = len(self.current_log)
        # sys.stdout.write(self.current_log)
        # sys.stdout.flush()
        # sys.stdout.write("\b"*100)
        # sys.stdout.flush()
        pass

    # def report(self):
    #     print("Score for last episode:", self.scores[-1])
