import torch.nn as nn
import torch
import torch.nn.functional as F


class DefectNetMsg(nn.Module):
    def __init__(self, args):
        super(DefectNetMsg, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        Dim = args.embedding_dim
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward(self, msg, code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_commit = x_msg
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out
