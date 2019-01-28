from tqdm import tqdm
import time

x = 100
every = 12

# for i in range(x):
#     time.sleep(0.01)
#     for ii in tqdm(range(every)):
#         time.sleep(0.01)

with tqdm(total=x) as pbar:
    #time.sleep(0.01)
    for ii in range(every):
        time.sleep(0.01)
        pbar.update(every)
