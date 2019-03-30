from IPython.display import Image
from IPython.display import display
import os
import os.path

def print_args(args):
    print('\n'.join(["{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)]))

def plot_results(imdir):
    images = [file for file in os.listdir(imdir) if os.path.splitext(file)[1] == '.png']
    num = len(images)
    fig = plt.figure()
    gs = GridSpec(num%2, min(num, 2))
    for i in range(num)
        ax = fig.add_subplot(gs[i%2, int((num/i)%2)])
        ax.plot()
