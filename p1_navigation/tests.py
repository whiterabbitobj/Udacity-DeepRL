# from tqdm import tqdm, trange
# import time
# import sys
# import progressbar
#
# # #
# # x = 500
# # every = 60
# # #
# #
# # for i in range(x):
# #     time.sleep(0.03)
# #     if i % every == 0:
# #         pbar = tqdm(total=100)
# #     pbar.update(100/every)
# #
# # for i in range(1,101):
# #     print(i)
# #     time.sleep(0.015)
# #     if i % 50 == 0:
# #         sys.stdout.flush()
#


import time
import progressbar

every = 6
x = 50

print("This is my main message")
# for i in progressbar.progressbar(range(100), redirect_stdout=True):
#     if i % 10 == 0:
#         print("some text", i)
#     time.sleep(0.1)
#

with progressbar.ProgressBar(max_value=every) as bar:
    for i in range(1,x+1):
        if i % every == 0:
            print("\nNew iteration!")
        time.sleep(0.1)

        bar.update(i%every+1)
