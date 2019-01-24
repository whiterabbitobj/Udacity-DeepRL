import torch

def activate_device(do_gpu):
    if do_gpu and torch.cuda.is_available():
        active_device = 'cuda'
    else:
        active_device = 'cpu'

    return active_device
