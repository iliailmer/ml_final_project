from swish import Swish
import torch


def replace_bn(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.ReLU:
            print('replaced: ', name, attr_str)
            setattr(m, attr_str, Swish())
