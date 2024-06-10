import random
import pickle
import numpy as np
import sys
import copy
from pose_env_base import Pose_Env_Base
from model import *
from scipy import stats
import time as timee
import torch
from replay_memory import *
from update import *
from utils import *
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from pathlib import Path
from torch.optim import Adam

import argparse
#parser = argparse.ArgumentParser()

torch.autograd.set_detect_anomaly(True)
#network
device = 'cpu'

#base policy
policy_net = Policy(5, 1, device).to(device)
#perturbation policy
cost_net = CPolicy(5, 1, device).to(device)

#target networks
cost_net_tar = CPolicy(5, 1, device).to(device)
policy_net_tar = Policy(5, 1, device).to(device)

#reward critic
value_net  = GValue(5, 1, device).to(device)
value_net_tar  = GValue(5, 1, device).to(device)

#cost critics
c1_net = GValue(5, 1, device).to(device)
c1_net_tar = GValue(5, 1, device).to(device)

optimizer1 = Adam(value_net.parameters(), lr=0.0001,
                                     weight_decay=1e-3)
optimizer3 = Adam(c1_net.parameters(), lr=0.0001,
                                     weight_decay=1e-3)
optimizer2 = Adam(policy_net.parameters(), lr=0.0005,
                                     weight_decay=1e-3)
optimizer_cost = Adam(cost_net.parameters(), lr=0.0003,
                                     weight_decay=1e-3)

#hard initialize target networks' para
hard_update(value_net_tar, value_net)
hard_update(c1_net_tar, c1_net)
hard_update(policy_net_tar, policy_net)
hard_update(cost_net_tar, cost_net)

#main body
def main():
    model_dir = Path('./models') / 'nd_seed'
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    models_dir = run_dir / 'models'
    os.makedirs(log_dir)
    os.makedirs(models_dir)
    logger = SummaryWriter(str(log_dir))
    
    Max_episodes = 50001
    mem = Memory()
    mem.clean()
    #set seed
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    random.seed(run_num)
    
    for episode in range(1, Max_episodes):
        if episode%10 == 0:
            #save models
            torch.save(policy_net.state_dict(), str(models_dir)+'/policy_'+str(episode)+'.pkl')
            torch.save(cost_net.state_dict(),   str(models_dir)+'/cost_'+str(episode)+'.pkl')
            
        env = Pose_Env_Base()
        env.seed(run_num)
        state = env.reset()
        
        total_ind_reward = 0
        total_glo_reward = 0
        total_cost   = 0
        
        while True:
            #state: 4*5*4
            ac_base, log_acs = policy_net(state, update=True)
            action = cost_net(state, ac_base, update=True) + ac_base
            action = torch.clamp(action, min=-1, max=1)

            xx = action.clone().detach().reshape(-1).numpy()
            new_state, reward, done, info = env.step(xx)
            
            ind_reward = np.mean(info['Reward'])
            glo_reward = info['Global_reward'][0]
            
            total_ind_reward += ind_reward
            total_glo_reward += glo_reward
            total_cost += info['cost']

            mem.push(state, action, ind_reward + glo_reward, info['cost'], total_cost, new_state, not done, log_acs)
            
            state = new_state
            if done:
                break
        
        print("Episode:", episode, "Reward:", total_ind_reward + total_glo_reward, "Ind_Reward:", total_ind_reward, "Glo_Reward:", total_glo_reward, "Cost:", total_cost)

        #train
        if episode % 10 == 0:
            mem.normalize()
            update_alg(mem, mem.memory, value_net, value_net_tar, optimizer1,\
                                        policy_net,policy_net_tar, optimizer2,\
                                        cost_net, cost_net_tar, optimizer_cost,\
                                        c1_net, c1_net_tar, optimizer3,\
                                        logger, run_num)
            mem.clean() 
        #log info
        logger.add_scalar('ind_rewards' , total_ind_reward, episode)
        logger.add_scalar('glo_rewards' , total_glo_reward, episode)
        logger.add_scalar('rewards' , total_glo_reward+total_ind_reward, episode)
        logger.add_scalar('cost', total_cost, episode)
        
    logger.close()
            
if __name__ == "__main__":
    main()
