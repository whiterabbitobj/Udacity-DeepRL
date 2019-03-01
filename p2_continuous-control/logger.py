import sys

class Logger:
    def __init__(self):
        self.current_log = ''
        self.full_log = ''
        self.rewards = []
        pass

    def add(self, log):
        self.current_log += str(log)

    def print(self):
        # flushlen = len(self.current_log)
        # sys.stdout.write(self.current_log)
        # sys.stdout.flush()
        # sys.stdout.write("\b"*100)
        # sys.stdout.flush()
