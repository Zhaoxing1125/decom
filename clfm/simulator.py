#env.py
import random
import numpy as np
import copy
import math
from objects import *
from scipy.stats import norm


class City():
    def __init__(self, Hexagon, Driver_dist, start_time):
        
        
        self.Hexagon = Hexagon
        self.Driver_dist = Driver_dist
        
        self.drivers = {}
        self.city_time = start_time
        
        self.nodes = {}
        self.construct_nodes()
        self.sample_orders()

        
    def construct_nodes(self):
        #nodes
        for id in self.Hexagon.keys():
            self.nodes[id] = Node(id, self.Hexagon[id]['neigh'])
        
        #drivers
        driver_id = 0
        for id, num in self.Driver_dist.items():
            #grid id has num avs
            for i in range(1, num+1):
                self.drivers[driver_id] = Driver(driver_id, self.city_time)
                
                self.drivers[driver_id].node = self.nodes[id]
                self.nodes[id].drivers[driver_id] = self.drivers[driver_id]
                
                driver_id += 1
                
    def get_global_state(self):
        state = {}
        
        #orders and cars
        for id, node in self.nodes.items():
            state[id] = [len(node.orders), len(node.drivers)]

        st_ = np.array(list(state.values()))

        #time step
        for id, st in zip(self.nodes.keys(), st_):
            state[id] = [st[0], st[1], self.city_time/36-1]

        return state
        
    def get_observation(self, id, state):
        #get observation
        obs = [state[id]]
        for xxx in self.nodes[id].neighbors[0]:
            if xxx in state:
                obs.append(state[xxx])
            else:
                obs.append([0, 0, self.city_time/36-1])
        
        return obs
            
                
    def increase_time(self):
        self.city_time += 1
        for driver in self.drivers.values():
            driver.city_time += 1

        for driver in self.drivers.values():
            driver.status_control_eachtime(self)
            
            
    def sample_orders(self):
        #clean history orders
        for node in self.nodes.values():
            node.orders = []
        
        #load new orders
        for id, node in self.nodes.items():
            if self.city_time in self.Hexagon[id]['orders']:
                orders_pool = self.Hexagon[id]['orders'][self.city_time]
            else:
                continue
            orders_cur = []
            
            for order in orders_pool:
                if order[1] < order[0]:
                    continue
                    
                #filter out invalid orders
                if order[2] not in self.nodes:
                    continue
                
                orders_cur.append(order)
            node.orders = copy.deepcopy(orders_cur)
                    
        
    def order_dispatch(self):
        fee_dict = {}
        fee_sum = 0
        gr_dict = {}
        
        orders_total_num = sum([len(xx.orders) for xx in self.nodes.values()])
        
        #only in current grid
        for node in self.nodes.values():
            nodefee_sum, nodefee_dict = node.order_dispatch()
            
            fee_sum += nodefee_sum
            fee_dict.update(nodefee_dict)
            gr_dict[node._index] = sum(list(nodefee_dict.values()))
            
        #apportion to neigh grid
        for node in self.nodes.values():
            for nei_id in node.neighbors[0]:
                if nei_id not in self.nodes:
                    continue
                if nei_id != -1:
                    nodefee_sum, nodefee_dict = node.utility_assign_orders_neighbor(self, nei_id)
            
                    fee_sum += nodefee_sum
                    fee_dict.update(nodefee_dict)
                    gr_dict[node._index] += sum(list(nodefee_dict.values()))
            
        orders_left_num = sum([len(xx.orders) for xx in self.nodes.values()])
            
        return fee_sum, fee_dict, 1-orders_left_num*1.0/(orders_total_num + 1), gr_dict
        
                
               
    def reposition(self, actions):
        action_gather = {}
        for id, node in self.nodes.items():
            action_gather[id] = 0
        
        for id, action in actions.items():
            
            driver = self.drivers[id]
                
            valid_neigh = [driver.node._index]
            valid_neigh.extend(driver.node.neighbors[0])
            
               
            dest = valid_neigh[action]
                
            if dest not in self.nodes:
                continue
            
            action_gather[dest] += 1

            driver.node.drivers.pop(id)
            driver.node = self.nodes[dest]
            self.nodes[dest].drivers[id] = driver

        return action_gather
            
    def get_ds_gap(self, state):
        st_ = np.array(list(state.values()))
        
        a = st_[:, 0].reshape(-1, 1) + 1e-6
        b = st_[:, 1].reshape(-1, 1) + 1e-6
        
        a = a/a.sum()
        b = b/b.sum()
        
        #calculate KL-d
        return (a * np.log(a/b)).sum()
            
        
    def step(self, actions):
        
        #reposition
        action_gather = self.reposition(actions)
        
        #dispatch orders
        reward, reward_dict, orr, gr_dict = self.order_dispatch()
        
        #increase one step
        self.increase_time()
        
        #load new orders
        self.sample_orders()
        
        state = self.get_global_state()
        kl_d  = self.get_ds_gap(state)
        
        return reward, reward_dict, orr, kl_d, np.var([x.reward for x in self.drivers.values()]), state, action_gather
        
        
        