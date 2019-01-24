import numpy as np
s = []
l = 10
for i in range(1,50+1):
    s.append(i)
    if i % l == 0:
        print(i)
        print(np.mean(s[-l:]))
    #print(i)
#print(np.mean(s))
