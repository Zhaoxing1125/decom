#main functions

import random
import pickle
import numpy as np
import sys
import copy
from simulator import *
from objects import *
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



#grid class
Hexagon = {}

#import grid and neighbors
#id:[neighbors' id]*6
def load_neigh_info():
    neigh_info = pickle.load(open('./neighbours', 'rb'))
    for id, neigh in neigh_info.items():
        if id not in Hexagon:
            #[[id...], [distance...]]
            Hexagon[id] = {}
            Hexagon[id]['neigh'] = neigh

#import order
#'eb9dd4095d9850e6287cefd813775a6c,35197,36907,353,447,3.54,0.053391736635926\n'
def load_order():
    orders = pickle.load(open('./orders_1week', 'rb'))
    for id in Hexagon.keys():
        if id in orders:
            #xxx = [copy.deepcopy(xx) for xx in orders[id] for i in range(1)]        #1 times orders
            Hexagon[id]['orders'] = orders[id]
        else:
            Hexagon[id]['orders'] = []

        
def sample_driver():
    #randomly initial drivers 
    driver = {}
    ddd = random.choices(list(Hexagon.keys()), k=500)
    for xxx in ddd:
        if xxx not in driver:
            driver[xxx] = 1
        else:
            driver[xxx] += 1
            
    return driver
        
def load():
    load_neigh_info()
    load_order()
    

def sample_actions(city, state, policy_net, cost_net):
    obs_pool = []
    neigh_obs_pool = []
    neigh_obs_dict = {}
    
    assert len([x for x in city.drivers.values() if x.onservice == False]) + len([x for x in city.drivers.values() if x.onservice == True]) == len(city.drivers)
    
    for id, driver in city.drivers.items():
        if driver.onservice is False:
            node_id = driver.node._index
            
            #get observation
            obs_pool.append(city.get_observation(node_id, state))
            
            if node_id in neigh_obs_dict:
                neigh_obs_pool.append(neigh_obs_dict[node_id])
                continue
            
            #get neighbors' observation
            neigh_obs = []
            for nei in driver.node.neighbors[0]:
                if nei in state:
                    neigh_obs.append(city.get_observation(nei, state))
                else:
                    neigh_obs.append([[0, 0, city.city_time/36-1]]*7)
            neigh_obs_pool.append(neigh_obs)
            
            
            neigh_obs_dict[node_id] = neigh_obs
    
    #generate action
    weights = step(obs_pool, neigh_obs_pool, policy_net, cost_net)
    
    action_selected = {}
    weight_generated = {}
    obs_calculated = {}
    avg_calculated = {}
    nei_obs_calculated = {}
    index = 0
    for id, driver in city.drivers.items():
        if driver.onservice is False:
            node_id = driver.node._index
            
            weight = weights[index].clone().detach().numpy()
            
            #calculate score
            e = abs(np.sum(weight * np.array(obs_pool[index]), axis=1)+1e-6)
            e /= e.sum()
                  
            
            #sample the destination
            action_selected[id] = np.random.choice(range(len(e)), p=e)
            
            #store
            weight_generated[id] = weights[index]
            obs_calculated[id] = obs_pool[index]
            nei_obs_calculated[id] = neigh_obs_pool[index]

            index += 1
            
            if node_id not in avg_calculated:
                avg_calculated[node_id] = weight
            

    return weight_generated, action_selected, obs_calculated, nei_obs_calculated, avg_calculated

torch.autograd.set_detect_anomaly(True)
#network
device = 'cpu'

#base policy
policy_net = Policy(2*7, 3, device).to(device)
#perturbation policy
cost_net = CPolicy(2*7, 3, device).to(device)

#target networks
cost_net_tar = CPolicy(2*7, 3, device).to(device)
policy_net_tar = Policy(2*7, 3, device).to(device) 

#reward critic
value_net  = GValue(2*7, 3, device).to(device)
value_net_tar  = GValue(2*7, 3, device).to(device)

#cost critics
c1_net = GValue(2*7, 3, device).to(device)
c2_net = GValue(2*7, 3, device).to(device)
c1_net_tar = GValue(2*7, 3, device).to(device)
c2_net_tar = GValue(2*7, 3, device).to(device)

optimizer1 = Adam(value_net.parameters(), lr=0.0001,
                                     weight_decay=1e-3)
optimizer3 = Adam(c1_net.parameters(), lr=0.0001,
                                     weight_decay=1e-3)
optimizer4 = Adam(c2_net.parameters(), lr=0.0001,
                                     weight_decay=1e-3)
optimizer2 = Adam(policy_net.parameters(), lr=0.001,
                                     weight_decay=1e-3)
optimizer_cost = Adam(cost_net.parameters(), lr=0.0003,
                                     weight_decay=1e-3)

#hard initialize target networks' para
hard_update(value_net_tar, value_net)
hard_update(c1_net_tar, c1_net)
hard_update(c2_net_tar, c2_net)
hard_update(policy_net_tar, policy_net)
hard_update(cost_net_tar, cost_net)


#main body
def main():
    model_dir = Path('./models') / 'od'
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
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    
    Max_episodes = 2001
    mem = Memory()
    
    
    #set seed
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)

    load()
    Driver = sample_driver()
    for episode in range(1, Max_episodes):
        
        
        if episode%10 == 0:
            #save models
            torch.save(value_net.state_dict(), './models/value_'+str(episode)+'.pkl')
            torch.save(policy_net.state_dict(), './models/policy_'+str(episode)+'.pkl')
            
        
        Min_steps = int(6*60/10)
        Max_steps = int(12*1*60/10)
        
        load()
        
        orders = []
        
        city = City(Hexagon, Driver, Min_steps)
        
        selected_cars = random.sample(list(city.drivers.keys()), k=200)
        mem.clean()
        
        total_kl = 0
        total_var = 0
        total_reward = 0
        total_orr = 0
        total_fee = 0
        
        state = city.get_global_state()

        acc_cost = [0, 0]

        for time in range(Min_steps, Max_steps):
                    
            #sample actions
            action_weights, action_selected, obs_calculated, nei_obs_calculated, avg_calculated = sample_actions(city, state, policy_net, cost_net)
            
            #take step
            reward, r_d, orr, kl_d, var, new_state, action_gather = city.step(action_selected)
            
            #calculate accumulated
            total_kl     += kl_d
            total_reward += reward
            total_orr    += orr
            total_var    += var
            total_fee    += sum(r_d.values())
            
            acc_cost[0] += kl_d
            acc_cost[1] += var
            
            #save buffer
            for car_id in action_weights.keys():
                if car_id in selected_cars:
                    
                    node = city.drivers[car_id].node
                    avg_acs = [action_weights[car_id].clone().detach().numpy()]
                    for nei in node.neighbors[0]:
                        if nei not in city.nodes:
                            avg_acs.append([0, 0, 0])
                            continue
                        if nei in avg_calculated:
                            avg_acs.append(avg_calculated[nei])
                        else:
                            avg_acs.append([0, 0, 0])
                    avg_acs = torch.tensor(avg_acs).mean(0)
                    if car_id not in r_d:
                        mem.push(car_id, list(state.values()), list(action_gather.values()), action_weights[car_id], avg_acs, reward, kl_d, var, acc_cost, obs_calculated[car_id], nei_obs_calculated[car_id], time-Min_steps)
                    else:
                        mem.push(car_id, list(state.values()), list(action_gather.values()), action_weights[car_id], avg_acs, reward, kl_d, var, acc_cost, obs_calculated[car_id], nei_obs_calculated[car_id], time-Min_steps)
                    
            state = new_state
            
            
            #log info
            Total_vehicle       = len(city.drivers)
            Online_vehicle      = len([xx for xx in city.drivers.values() if xx.online is True])
            Offline_vehicle      = len([xx for xx in city.drivers.values() if xx.online is False])
            Onservice_vehicle   = len([xx for xx in city.drivers.values() if xx.onservice is True])
            
            
            assert Total_vehicle == Online_vehicle + Offline_vehicle
            
            print("===============================================================================")
            print("Time ", time, "Reward: ", reward, "KL: ", kl_d, "Orr: ", orr, "Var:", var)
            print("T_V: ", Total_vehicle, "Of_V: ", Offline_vehicle, "OS_V: ", Onservice_vehicle)

        
        print("Episode:", episode, "Reward:", total_reward, "KL:", total_kl, "Var:", total_var, "Orr:", total_orr)

        #train
        mem.normalize()
        update_alg(mem, mem.memory, value_net, optimizer1, policy_net, optimizer2, logger, value_net_tar, policy_net_tar, c1_net, c2_net, c1_net_tar, c2_net_tar, optimizer3, optimizer4, cost_net, cost_net_tar, optimizer_cost)
        
        
        #log info
        logger.add_scalar('episode_rewards' , total_reward, episode)
        logger.add_scalar('episode_kl', total_kl, episode)
        logger.add_scalar('episode_var', total_var, episode)
        logger.add_scalar('episode_orr', total_orr, episode)
        logger.add_scalar('episode_fee', total_fee, episode)
        
        
    logger.close()
            
if __name__ == "__main__":
    main()
