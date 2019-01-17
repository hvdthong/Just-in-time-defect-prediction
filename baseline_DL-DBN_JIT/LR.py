import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        # self.fc = nn.Linear(input_size, 128)
        # self.fc1 = nn.Linear(128, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, num_classes)

        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_size):
        # out = self.fc(input_size)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)

        out = self.fc(input_size)
        out = self.sigmoid(out).squeeze(1)
        return out
