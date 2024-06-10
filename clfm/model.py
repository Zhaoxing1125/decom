import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from exploration import OrnsteinUhlenbeckProcess as OUP
from torch.distributions import Normal



import numpy as np

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(Policy, self).__init__()
        
        self.num_inputs = num_inputs
        self.device = device
        
        
        self.l1 = nn.Linear(3, 2)
        self.l11 = nn.Linear(2, 2)
        self.l2 = nn.Linear(3, 2)
        self.l21 = nn.Linear(2, 2)
        self.l3 = nn.Linear(4, 3)
        
        self.oup = OUP(size=2, theta=0.15, mu=0.0, sigma=0.2)
        
    def forward(self, x, update=False):
        x = np.array(x).reshape(-1, 21)
        x = torch.tensor(x, dtype=torch.float)
        
        x1 = x[:, :3].reshape(-1, 3)#torch.cat((x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)), dim=1)
        x2 = x[:, 3:]
        
        x1 = F.elu(self.l1(x1))
        x1 = F.elu(self.l11(x1))
        
        x2 = self.l2(x2.reshape(-1, 6, 3))
        x2 = F.elu(x2)
        x2 = F.elu(self.l21(x2))
        x2 = x2.mean(dim=1)
        
        x3 = F.relu(self.l3(torch.cat((x1, x2), dim=1)))

        noise = Normal(torch.tensor([0, 0, 0], dtype=torch.float), torch.tensor([1, 1, 1], dtype=torch.float))

        z = noise.sample()
        action = torch.tanh(x3 + z)

        dist = Normal(x3, torch.tensor([1, 1, 1]).float())
        min_Val = torch.tensor(1e-7).float()
        log_prob = (dist.log_prob(x3 + z) - torch.log(1 - action.pow(2) + min_Val)).sum(1).reshape(-1, 1)
        
        #x3 = x3 + torch.tensor(self.oup.sample(), dtype=torch.float)

        if not update:
            return action
        else:
            return action, log_prob
            
class CPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(CPolicy, self).__init__()
        
        self.num_inputs = num_inputs
        self.device = device
        
        
        self.l1 = nn.Linear(3, 2)
        self.l11 = nn.Linear(2, 2)
        self.l2 = nn.Linear(3, 2)
        self.l21 = nn.Linear(2, 2)
        self.l3 = nn.Linear(4, 3)
        
        self.oup = OUP(size=3, theta=0.15, mu=0.0, sigma=0.2)
        
        self.l4 = nn.Linear(3, 3)
        self.l5 = nn.Linear(3, 3)
        
        self.l6 = nn.Linear(3, 3)
        self.l7 = nn.Linear(3, 3)
        self.l8 = nn.Linear(9, 3)
        
    def forward(self, x, ac, ac_nei, update=False):
        x = np.array(x).reshape(-1, 21)
        x = torch.tensor(x, dtype=torch.float)
        
        x1 = x[:, :3].reshape(-1, 3)#torch.cat((x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)), dim=1)
        x2 = x[:, 3:]
        
        x1 = F.elu(self.l1(x1))
        x1 = F.elu(self.l11(x1))
        
        x2 = self.l2(x2.reshape(-1, 6, 3))
        x2 = F.elu(x2)
        x2 = F.elu(self.l21(x2))
        x2 = x2.mean(dim=1)
        
        x3 = F.elu(self.l3(torch.cat((x1, x2), dim=1)))
        
        x4 = F.elu(self.l4(ac))
        x5 = F.elu(self.l5(x4))
        
        x6 = F.elu(self.l6(ac_nei))
        x6 = F.elu(self.l7(x6))
        x6 = x6.mean(1).reshape(-1, 3)
        
        x6 = F.relu(self.l8(torch.cat((x3, x5, x6), dim=1)))

        if not update:
            x6 = x6 + torch.tensor(self.oup.sample(), dtype=torch.float)

        return x6
             
class Value(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(Value, self).__init__()
        
        self.num_inputs = num_inputs
        self.device = device
        
        
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 2)
        self.l3 = nn.Linear(4, 2)
        
        self.l4 = nn.Linear(2, 2)
        self.l5 = nn.Linear(6, 3)
        self.l7 = nn.Linear(3, 1)
        
        self.l6 = nn.Linear(2, 2)
        
        
    def forward(self, x, ac, ac_avg):
        x = np.array(x).reshape(-1, 14)
        x = torch.tensor(x, dtype=torch.float)
        
        x1 = torch.cat((x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)), dim=1)
        x2 = x[:, 2:]
        
        x1 = self.l1(x1)
        x1 = F.elu(x1)
        
        x2 = self.l2(x2.reshape(-1, 6, 2))
        x2 = F.elu(x2)
        x2 = x2.mean(dim=1)
        
        x3 = self.l3(torch.cat((x1, x2), dim=1))
        x3 = F.elu(x3)
        
        ac = self.l4(ac)
        ac = F.elu(ac)
        
        ac_avg = self.l4(ac_avg)
        ac_avg = F.elu(ac_avg)
        
        xx = self.l5(torch.cat((x3, ac, ac_avg), dim=1))
        xx = F.elu(xx)
        xx = self.l7(xx)
        
        return xx


class GValue(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(GValue, self).__init__()
        
        self.num_inputs = num_inputs
        self.device = device
        
        
        self.l1 = nn.Linear(3, 2)
        self.l2 = nn.Linear(2, 1)
        self.l3 = nn.Linear(1, 1)

        self.l4 = nn.Linear(3, 2)
        self.l5 = nn.Linear(5, 3)
        self.l7 = nn.Linear(3, 1)
        
        self.l6 = nn.Linear(3, 2)
        
        
    def forward(self, state, ac, ac_avg):
        #state:  global state : 100*2
        #action: global action: 100*1

        state_len = len(state[0])

        
        x1 = F.elu(self.l1(state))
        #x1 = x1[:, :, 0].reshape(-1, state_len, 1) - x1[:, :, 1].reshape(-1, state_len, 1)
        x1 = F.elu(self.l2(x1))
        
        x1 = F.elu(self.l3(x1.mean(1)))

        ac = self.l4(ac)
        ac = F.elu(ac)
        
        ac_avg = self.l6(ac_avg)
        ac_avg = F.elu(ac_avg)
        
        xx = self.l5(torch.cat((x1, ac, ac_avg), dim=1))
        xx = F.elu(xx)
        xx = self.l7(xx)

        return xx
