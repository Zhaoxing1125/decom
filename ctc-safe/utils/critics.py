import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class Critic(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
        """
        super(Critic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()

        self.nn_test = {}

        # iterate over agents
        i = 0
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)         
            
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2*hidden_dim, hidden_dim))
            critic.add_module('critic_nl1', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, int(hidden_dim/4)))
            critic.add_module('critic_n2', nn.LeakyReLU())
            critic.add_module('critic_fc3', nn.Linear(int(hidden_dim/4), int(hidden_dim/16)))
            critic.add_module('critic_n4', nn.LeakyReLU())
            critic.add_module('critic_fc5', nn.Linear(int(hidden_dim/16), 1))
            self.critics.append(critic)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            
            critic_in = torch.cat((sa_encodings[i], torch.stack([x.clone() for j, x in enumerate(sa_encodings) if j != i]).mean(0).view(sa_encodings[i].shape[0], 128)), dim=1)
            q = self.critics[a_i](critic_in)

            if return_q:
                agent_rets.append(q)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

class cost_Critic(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
        """
        super(cost_Critic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()

        self.nn_test = {}

        # iterate over agents
        i = 0
        for sdim, adim in sa_sizes:
            idim = sdim + adim + 3 # 3 for one-hot index
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            encoder.add_module('enc_fc2', nn.Linear(hidden_dim, hidden_dim))
            encoder.add_module('enc_nl2', nn.LeakyReLU())
            self.critic_encoders.append(encoder)         
            
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2*hidden_dim, hidden_dim))
            critic.add_module('critic_nl1', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, int(hidden_dim/4)))
            critic.add_module('critic_n2', nn.LeakyReLU())
            critic.add_module('critic_fc3', nn.Linear(int(hidden_dim/4), 1))
            self.critics.append(critic)

    def forward(self, inps, index, agents=None, return_q=True, return_all_q=False,
                regularize=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        inps = [torch.cat((s, a, torch.tensor(index, dtype=torch.float).repeat(len(s), 1)), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            
            critic_in = torch.cat((sa_encodings[i], torch.stack([x.clone() for j, x in enumerate(sa_encodings) if j != i]).mean(0).view(sa_encodings[i].shape[0], 128)), dim=1)
            q = self.critics[a_i](critic_in)

            if return_q:
                agent_rets.append(q)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
