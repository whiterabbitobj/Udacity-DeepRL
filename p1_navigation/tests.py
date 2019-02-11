from collections import namedtuple, deque
import random

x = deque(maxlen=20)
for i in range(10):
    x.append(i)

print(x)
print(x[-1])
for i in range(10,20):
    x.append(i)
print(x)
print(x[-1])
for i in range(20,30):
    x.append(i)
print(x)
print(x[-1])
for i in range(30,40):
    x.append(i)
print(x)
print(x[-1])




y = deque([0,0,0,0],maxlen=4)
print(y)
y.append(1)
print(y)
