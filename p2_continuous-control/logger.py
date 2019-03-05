import sys
import numpy as np
import time

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

    # def report(self):
    #     print("Score for last episode:", self.scores[-1])

    def start_clock(self):
        t = time.localtime()
        print("Start time: {}:{}:{}".format(t.tm_hour, t.tm_min, t.tm_sec))
        self.start_time = time.time()

    def step(self, eps):
        self._update_score()
        print("Episode {}/{}... RUNTIME: {}".format(eps, self.max_eps, self._runtime()))
        self._reset_rewards()

    def _runtime(self):
        m, s = divmod(time.time() - self.start_time, 60)
        h, m = divmod(m, 60)
        return "{}h{}m{}s".format(int(h), int(m), int(s))

    def _update_score(self):
        score = self.rewards.mean()
        print("...Episode return: ", score)
        self.scores.append(score)

    def print(self):
        # flushlen = len(self.current_log)
        # sys.stdout.write(self.current_log)
        # sys.stdout.flush()
        # sys.stdout.write("\b"*100)
        # sys.stdout.flush()
        pass

    def _reset_rewards(self):
        self.rewards = np.zeros(self.agent_count)
