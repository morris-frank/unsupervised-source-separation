from torch import nn
from ...utils import clean_init_args


class MONet(nn.Module):
    def __init__(self):
        super(MONet, self).__init__()
        self.params = clean_init_args(locals().copy())
