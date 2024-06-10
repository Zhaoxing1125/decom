import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample
from torch.distributions import Normal
import sys
from utils.exploration import OrnsteinUhlenbeckProcess as OUP

        
class Cont_BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Cont_BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/16))
        self.fc3 = nn.Linear(int(hidden_dim/16),  out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        xx = self.fc1(inp)
        h1 = self.nonlin(xx)
        h1 = self.fc2(h1)
        h2 = self.nonlin(h1)
        h2 = self.fc3(h2)
        action = F.tanh(h2)
        return action    
class ContPolicy(Cont_BasePolicy):
    """
    Policy Network for continuous action spaces
    """
    def __init__(self, *args, **kwargs):
        super(ContPolicy, self).__init__(*args, **kwargs)

        self.oup = OUP(size=2, theta=0.15, mu=0.0, sigma=0.2)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False, update_policy=False):
        action = super(ContPolicy, self).forward(obs)

        on_gpu = next(self.parameters()).is_cuda
        
        if on_gpu:
            noise = torch.tensor(self.oup.sample(), dtype=torch.float).cuda(device='cuda:1')
        else:
            noise = torch.tensor(self.oup.sample(), dtype=torch.float)

        if not sample:
            noise = 0

        action = action.clone() + noise
        action = torch.clamp(action, -1, 1)
        
        rets = [action]

        if return_all_probs:
            pass

        if return_log_pi:
            pass

        if regularize:
            pass
        if return_entropy:
            pass

        if len(rets) == 1:
            return rets[0]
        return rets


class Cont_BasePolicy_cost(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, action_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Cont_BasePolicy_cost, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, int(hidden_dim/32))
        self.fc2 = nn.Linear(int(hidden_dim/32), int(hidden_dim/32))
        self.fc3 = nn.Linear(action_dim, int(hidden_dim/32))
        self.fc4 = nn.Linear(int(hidden_dim/32),  int(hidden_dim/32))

        self.fc5 = nn.Linear(int(hidden_dim/16),  out_dim)
        self.nonlin = nonlin

    def forward(self, X, ac):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        
        h1 = self.nonlin(self.fc1(inp))
        h1 = self.nonlin(self.fc2(h1))

        h2 = self.nonlin(self.fc3(ac))
        h2 = self.nonlin(self.fc4(h2))

        h2 = self.fc5(torch.cat((h1, h2), dim=1))
        action = F.tanh(h2)
        return action    
class ContPolicy_cost(Cont_BasePolicy_cost):
    """
    Policy Network for continuous action spaces
    """
    def __init__(self, *args, **kwargs):
        super(ContPolicy_cost, self).__init__(*args, **kwargs)

        self.oup = OUP(size=2, theta=0.15, mu=0.0, sigma=0.2)

    def forward(self, obs, ac, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False, update_policy=False):
        action = super(ContPolicy_cost, self).forward(obs, ac)

        on_gpu = next(self.parameters()).is_cuda
        
        if on_gpu:
            noise = torch.tensor(self.oup.sample(), dtype=torch.float).cuda(device='cuda:1')
        else:
            noise = torch.tensor(self.oup.sample(), dtype=torch.float)

        if not sample:
            noise = 0

        action = action.clone() + noise
        action = torch.clamp(action, -1, 1)
        
        rets = [action]

        if return_all_probs:
            pass

        if return_log_pi:
            pass

        if regularize:
            pass
        if return_entropy:
            pass

        if len(rets) == 1:
            return rets[0]
        return rets