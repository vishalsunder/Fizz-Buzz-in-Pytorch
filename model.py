import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.layer1 = nn.Linear(args.num_bits,args.num_units)
        self.layer2 = nn.Linear(args.num_units,args.num_units)
        self.layer3 = nn.Linear(args.num_units,4)
        self.dropout = nn.Dropout(0.5)        

    def forward(self,input):
        logits = self.layer3(self.dropout(F.relu(self.layer2(self.dropout(F.relu(self.layer1(input)))))))
        return F.softmax(logits),logits
