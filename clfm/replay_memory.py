import random
from collections import namedtuple
import numpy as np

                                       
class Memory(object):
    def __init__(self):
        self.memory = {}
        self.niter_p = 0
        self.niter_q = 0

    def push(self, id, state, action_gather, action, avg_acs, reward, kl, var, acc_cost, obs, neigh_obs, t):
        if id not in self.memory:
            self.memory[id] = [[state, action_gather, action, avg_acs, reward, kl, var, acc_cost, obs, neigh_obs, t]]
        else:
            self.memory[id].append([state, action_gather, action, avg_acs, reward, kl, var, acc_cost, obs, neigh_obs, t])

    def normalize(self):
        reward = []
        for _, items in self.memory.items():
            for item in items:
                reward.append(item[4])
        #normalize reward
        r_mean = np.mean(reward)
        r_var  = np.var(reward)

        for _, items in self.memory.items():
            for item in items:
                item[4] = (item[4] - r_mean) / (r_var+1e-6)

    def clean(self):
        for xx in self.memory.keys():
            self.memory[xx] = {}
            
        self.memory = {}
        
        
        
    
