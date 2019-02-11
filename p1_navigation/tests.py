from collections import namedtuple, deque
import random

class stuff:
    def __init__(self):
        self.m = []
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
    # def _memory(self, *args):
    #     return namedtuple("memory", field_names=['state','action','reward','next_state'])

    def add(self, s, a, r, ns):
        # self.m.append(self._memory(s, a, r, ns))
        self.m.append(self.memory(s,a,r,ns))
    def print(self):
        for i in self.m:
            print(i.state, i.action)
c = stuff()

c.add('1','0','-1','2')
c.add('2','1','1','1')
c.add('5','3','0','7')

c.print()

a = namedtuple("memory", field_names=['state','action','reward','next_state'])
x = memory('1','0','-1','2')
y = memory('2','1','1','1')
z = memory('5','3','0','7')

a = deque(maxlen=3)
a.append(x)
a.append(y)
a.append(z)

print(a)

b = random.sample(a, 2)

print(b)

c = zip(*b)

for i in c:
    print(i)


d = memory(*zip(*b))
print(d)
print(d.state)
