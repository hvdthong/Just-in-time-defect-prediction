import torch.nn as nn
import torch
import torch.nn.functional as F


class DefectNet(nn.Module):
    def __init__(self, args):
        super(DefectNet, self).__init__()
        self.args = args