import random
from collections import namedtuple
import numpy as np
                                       
class Memory(object):
    def __init__(self):
        self.memory = []

        self.niter_p = 0
        self.niter_q = 0
        self.niter_p_cost = 0

    def push(self, s, a, r, c, acc_c, ns, done, log_acs):
        self.memory.append([s, a, r, c, acc_c, ns, done, log_acs])

    def normalize(self):
        reward = []
        for item in self.memory:
            reward.append(item[2])
        #normalize reward
        r_mean = np.mean(reward)
        r_var  = np.var(reward)

        for item in self.memory:
            item[2] = (item[2] - r_mean) / (r_var+1e-6)

    def clean(self):
        self.memory = []
        
        
        
        
    
