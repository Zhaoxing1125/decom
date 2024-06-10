# -*- coding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod
import random
import math



class Node(object):
    __slots__ = ('neighbors', 
                 '_index', 
                 'orders', 
                 'drivers')
    
    def __init__(self, index, neighbors):
        # private
        self._index = index   # unique node index.

        # public
        self.neighbors = neighbors  # a list of nodes that neighboring the Nodes
        self.orders = []     # a list of orders
        self.drivers = {}    # a dictionary of driver objects contained in this node
        
        
    def clean_node(self):
        self.orders = []
        self.drivers = {}
        
    def add_driver(self, driver_id, driver):
        self.drivers[driver_id] = driver
        

    def order_dispatch(self):
        
        fee_dict = {}
        fee_sum = 0
        served_orders = []
        
        for order_id, order in enumerate(self.orders):
            for driver_id, driver in self.drivers.items():
                if driver.onservice is False and \
                   driver.online    is True :
                    
                    duration = order[1]/60/10
                    
                    #order information
                    en_time     = math.ceil(duration) if int(duration) > driver.city_time else math.ceil(duration)+1
                    destination = order[2]
                    fee         = order[3]
                    distance    = order[4]
                    
                    
                    order[1] = en_time
                    
                    #take order
                    driver.take_order(order)
                    self.drivers.pop(driver_id)
                    
                    fee_dict[driver_id] = fee
                    fee_sum += 1
                    driver.reward += 1
                    
                    served_orders.append(order_id)
                    
                    break
                        
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_orders]
        
        return fee_sum, fee_dict

    def utility_assign_orders_neighbor(self, city, neighbor_node):
        
        fee_dict = {}
        fee_sum = 0
        served_orders = []
        
        for order_id, order in enumerate(self.orders):
            
            for driver_id, driver in city.nodes[neighbor_node].drivers.items():
                if driver.onservice is False and \
                   driver.online    is True:
                    
                    duration = order[1]/60/10
                    
                    #order information
                    en_time     = math.ceil(duration) if int(duration) > driver.city_time else math.ceil(duration)+1
                    destination = order[2]
                    fee         = order[3]
                    distance    = order[4]
                    
                    order[1] = en_time
                    
                    
                    driver.take_order(order)
                    city.nodes[neighbor_node].drivers.pop(driver_id)
                    
                    fee_dict[driver_id] = fee
                    fee_sum += 1
                    driver.reward += 1
                    
                    served_orders.append(order_id)
                    
                    
                    break       
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_orders]
        
        return fee_sum, fee_dict

    
                            
class Driver(object):
    __slots__ = ( 'online', 'onservice', 'order', 'node', 'city_time', '_driver_id',
                 'reward')

    def __init__(self, driver_id, city_time):
        self.online=True
        self.onservice= False

        self.order = None     # the order this driver is serving
        self.node = None      # the node that contain this driver.
        self.city_time = city_time  # track the current system time
        

        self.reward = 0         # accumulative order num
        
        # private
        self._driver_id = driver_id  # unique driver id.
    

    def set_order_start(self, order):
        self.order = order
    
    def set_order_finish(self):
        self.order = None
        self.onservice = False
        self.online=True

    def get_driver_id(self):
        return self._driver_id

    def update_city_time(self):
        self.city_time += 1

    def set_city_time(self, city_time):
        self.city_time = city_time


    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
        """
        assert self.online is True
        assert self.onservice is False
        self.set_order_start(order)
        self.onservice = True
        
        

    def status_control_eachtime(self, city):
        assert self.city_time == city.city_time
        if self.onservice is True:
            order_end_time = self.order[1]
            
            #check whether order finishes
            if self.city_time == order_end_time:
                self.node = city.nodes[self.order[2]]
                self.set_order_finish()
                
                self.node.add_driver(self._driver_id, self)
            elif self.city_time < order_end_time:
                pass
            else:
                print(self._driver_id)
                print(self.order)
                raise ValueError('Driver: status_control_eachtime(): order end time less than city time')
                
                        


