import sys

class Logger:
    def __init__(self, env):
        self.current_log = ''
        self.full_log = ''
        self.agent_count = env.agent_count
        self.scores = []
        self._reset_rewards()


    def add(self, log):
        self.current_log += str(log)

    def log_score(self):
        self.scores.append(logger.score())

    def report(self):
        print("Score for last episode:", self.scores[-1])

    def score(self):
        score = self.rewards.mean()
        self.scores.append(score)
        self._reset_rewards()

    def print(self):
        # flushlen = len(self.current_log)
        # sys.stdout.write(self.current_log)
        # sys.stdout.flush()
        # sys.stdout.write("\b"*100)
        # sys.stdout.flush()
        pass

    def _reset_rewards(self):
        self.rewards = np.zeros(self.agent_count)
