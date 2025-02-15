import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class Attention(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
        """
    def __init__(self, out_features=64, in_features=3):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_q = Parameter(th.FloatTensor(out_features, in_features))
        self.weight_k = Parameter(th.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_q) 
        nn.init.kaiming_uniform_(self.weight_k) 

    def forward(self, x):
        v  = x["value"]
        target = x["target"]
        # D = target.shape[0] # [-1]が正しかったりする？ 
        # B = v.shape[0]
        # print(v.shape) # B,S,L # torch.Size([440, 16, 64])
        # print(target.shape) # B,3 # torch.Size([440, 3])

        q = F.linear(target, self.weight_q) # B,64
        q = q.unsqueeze(1) # B,1,64
        D = q.shape[-1]
        k = F.linear(target, self.weight_k) # B,64
        k = k.unsqueeze(2) # B,64,1

        # print(q.shape) # torch.Size([440, 1, 64])

        r = th.bmm(q,k)/math.sqrt(D)
        # print(r.shape) # torch.Size([440, 1, 1])
        r = r.squeeze()
        # print(r.shape) # torch.Size([440])
        r = F.softmax(r, dim=0)
        output = {}
        output["w_mean"] = th.sum(r[:,None,None] * v * v,dim=0)
        output["weight"] = r
        
        return output

class Pos2Weight(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
        """
    def __init__(self, out_features=1, in_features=3, channel=64, config={}):
        super(Pos2Weight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channel = channel

        modules = []
        modules.extend([nn.Linear(in_features,channel),
                        nn.LayerNorm(channel),
                        nn.ReLU()])
        for l in range(config["hlayers_p2w"]): 
            modules.extend([nn.Linear(channel,channel),
                            nn.LayerNorm(channel),
                            nn.ReLU()])
        modules.extend([nn.Linear(channel,out_features),
                        nn.Sigmoid()
                        ])
        self.pos2weight = nn.Sequential(*modules)
        
        if config["temp_learnable"]:
            self.temperature = nn.parameter.Parameter(th.tensor([config["temp_init"]]))
        else:
            self.temperature = config["temp_init"]
        # self.temperature = nn.parameter.Parameter(th.tensor([temperature]))

    def forward(self, x):
        v  = x["value"]
        pos = x["target"]
        # print(pos.shape) # torch.Size([361, 3])
        w = self.pos2weight(pos)
        # print(w.shape) # torch.Size([361, 1])
        w = w.squeeze()
        w = F.softmax(w/self.temperature, dim=0)
        output = {}
        output["w_mean"] = th.sum(w[:,None,None] * v,dim=0)
        output["weight"] = w
        # print(f'Temp: {self.temperature}')
        
        return output
