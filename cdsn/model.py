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
        
        self.device = device
        
        #4*5*4
        
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 2)
        self.l3 = nn.Linear(10, 5)
        self.l4 = nn.Linear(5, 1)
        
        #self.oup = OUP(size=1, theta=0.15, mu=0.0, sigma=0.5)
        
    def forward(self, x, update=False):
        x1 = torch.tensor(x, dtype=torch.float).reshape(-1, 4, 5, 4)
        
        x2 = F.tanh(self.l1(x1)) #4*5*4
        x3 = F.elu(self.l2(x2)) #4*5*2
        x3 = x3.reshape(-1, 4, 10)
        x4 = F.tanh(self.l3(x3)) #4*5
        #x5 = torch.mean(x4, dim=2).reshape(-1, 4, 1, 4)
        x3 = self.l4(x4).reshape(-1, 4, 1)

        noise = Normal(torch.tensor([0, 0, 0, 0], dtype=torch.float), torch.tensor([1, 1, 1, 1], dtype=torch.float))

        a_dim = len(x3)

        z = noise.sample((a_dim, 1))
        action = torch.tanh(x3 + z.reshape(a_dim, 4, 1))

        dist = Normal(x3, torch.tensor([1, 1, 1, 1]).float())
        min_Val = torch.tensor(1e-7).float()
        log_prob = (dist.log_prob(x3 + z) - torch.log(1 - action.pow(2) + min_Val)).sum(2).mean(1).reshape(-1, 1)
        
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
        
        
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 2)
        self.l3 = nn.Linear(4, 4)
        self.l4 = nn.Linear(4, 8)
        
        self.l5 = nn.Linear(18, 8)
        
        self.oup = OUP(size=1, theta=0.15, mu=0.0, sigma=0.2)
        
        self.l6 = nn.Linear(8, 1)
        
        
    def forward(self, x, ac, update=False):
        x1 = torch.tensor(x, dtype=torch.float).reshape(-1, 4, 5, 4)
        
        x2 = F.tanh(self.l1(x1)) #4*5*4
        x3 = F.elu(self.l2(x2)) #4*5*2
        x3 = x3.reshape(-1, 4, 10)
        
        ac1 = F.elu(self.l3(torch.transpose(ac.reshape(-1, 4, 1).repeat(1, 1, 4), 1, 2)))
        ac2 = F.elu(self.l4(ac1)).reshape(-1, 4, 8) #1*4
        
        y = torch.cat((x3, ac2), dim=2)
        
        x4 = F.tanh(self.l5(y)) #4*8
        z = self.l6(x4).reshape(-1, 4, 1)
        
        
        z = z + torch.tensor(self.oup.sample(), dtype=torch.float)
            
        z = torch.tanh(z)
        
        return z
             
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
        
        # 4*5*4
        # 4*1
        
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 1)

        self.l4 = nn.Linear(1, 2)
        self.l5 = nn.Linear(2, 1)
        
        self.l6 = nn.Linear(8, 4)
        self.l7 = nn.Linear(4, 1)
        
        
    def forward(self, state, ac):
        x = F.elu(self.l1(state))
        x = F.elu(self.l2(x))
        x = torch.mean(x, dim=2).reshape(-1, 4, 4)
        x = F.elu(self.l3(x)).reshape(-1, 4)
        #out:d*4
        #print(state.shape, x.shape)
        #print(ac.shape)
        y = F.elu(self.l4(ac))
        y = F.elu(self.l5(y)).reshape(-1, 4)
        #out:d*4
        #print(ac.shape, y.shape)
        z = torch.cat((x, y), dim=1)
        #out:d*8
        
        xx = self.l7(F.elu(self.l6(z)))
        
        return xx
