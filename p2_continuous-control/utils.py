import os.path
import re
import time

import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def print_bracketing(info=None, do_upper=True, do_lower=True):
    mult = 50
    if type(info) is not list and info is not None:
        mult = max(mult, len(info))
        info = [info]
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    if do_upper: print(upper)
    if info is not None:
        for line in info:
            print(line.center(mult))
    if do_lower: print(lower)
    return
