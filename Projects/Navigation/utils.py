# -*- coding: utf-8 -*-
import os.path



def print_bracketing(info=None, do_upper=True, do_lower=True):
    """
    Formats a provided statement (INFO) for printing to the cmdline. If provided
    a list, will print each element to a new line, and if provided a string,
    will print a single line.
    """
    mult = 50
    if type(info) is not list and info is not None:
        mult = max(mult, len(info))
        info = [info]
    bracket = "#"
    upper = ("{0}\n{1}{2}{1}".format(bracket*mult, bracket, " "*(mult-2)))
    lower = ("{1}{2}{1}\n{0}".format(bracket*mult, bracket, " "*(mult-2)))
    if do_upper: print(upper)
    if info: print('\n'.join([line.center(mult) for line in info]))
    if do_lower: print(lower)
    return

def check_dir(dir):
    """
    Creates requested directory if it doesn't yet exist.
    """

    if not os.path.isdir(dir):
        os.makedirs(dir)
