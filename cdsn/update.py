import torch
import random
import math
import numpy as np
from torch.autograd import Variable

from utils import *
import random


def update_alg(Mem, mem, value_net, value_net_tar, optimizer1,\
                                        policy_net,policy_net_tar, optimizer2,\
                                        cost_net, cost_net_tar, optimizer_cost,\
                                        c1_net, c1_net_tar, optimizer3,\
                                        logger, run_num):
    
    #update critics
    random.seed(run_num)
    for _ in range(10):
        step_list = random.choices(range(len(mem)-1), k=int(len(mem) * 0.3))
        
        reward       = torch.zeros((len(step_list), 1))
        cost         = torch.zeros((len(step_list), 1))
        done         = torch.zeros((len(step_list), 1))
        global_state = torch.zeros((len(step_list), 4, 5, 4))
        global_next_state = torch.zeros((len(step_list), 4, 5, 4))
        global_acs   = torch.zeros((len(step_list), 4, 1))
            
        for index, i  in enumerate(step_list):
            reward[index] = mem[i][2]
            cost[index] = mem[i][3]
            done[index]   = mem[i][6]
            global_state[index] = torch.from_numpy(np.array(mem[i][0]))
            global_next_state[index] = torch.from_numpy(np.array(mem[i+1][0]))
            global_acs[index]   = mem[i][1]
                
        curr_q = value_net(global_state, global_acs)
        next_acs = sample_action(global_next_state, policy_net_tar, cost_net_tar, update=False)
        next_q = value_net_tar(global_next_state,  next_acs.clone().detach())
        q_ind_loss  = 0
        for i, cq, nq in zip(range(len(step_list)), curr_q, next_q):
            q_ind_loss += (cq - reward[i] - nq * 0.99 * done[i] )**2
        q_loss = q_ind_loss / (1+len(step_list))
        optimizer1.zero_grad()
        q_loss.backward(retain_graph=True, inputs=list(value_net.parameters()))
        grad_norm_q = torch.nn.utils.clip_grad_norm(value_net.parameters(), 0.5)
        optimizer1.step()
        soft_update(value_net_tar, value_net, 0.005)
        logger.add_scalar('q_loss/loss', q_loss, Mem.niter_q)
        logger.add_scalar('q_loss/grad', grad_norm_q, Mem.niter_q)
        
        curr_q = c1_net(global_state, global_acs)
        #next_acs = sample_action(global_next_state, policy_net_tar, update=False)
        next_q = c1_net_tar(global_next_state,  next_acs.clone().detach())
        q_ind_loss  = 0
        for i, cq, nq in zip(range(len(step_list)), curr_q, next_q):
            q_ind_loss += (cq - cost[i] - nq * done[i] )**2
        q_loss = q_ind_loss / (1+len(step_list))
        optimizer3.zero_grad()
        q_loss.backward(retain_graph=True, inputs=list(c1_net.parameters()))
        grad_norm_q = torch.nn.utils.clip_grad_norm(c1_net.parameters(), 0.5)
        optimizer3.step()
        soft_update(c1_net_tar, c1_net, 0.005)
        logger.add_scalar('c_q_loss/loss', q_loss, Mem.niter_q)
        logger.add_scalar('c_q_loss/grad', grad_norm_q, Mem.niter_q)
        
        Mem.niter_q += 1
    print("critic update successful!")
    
    #update base policy
    for _ in range(1):
        p_loss = 0
        for ii in range(10):
            step_list = random.choices(range(len(mem)), k=int(len(mem) * 0.3))
            global_state = torch.zeros((len(step_list), 4, 5, 4))
            global_acs   = torch.zeros((len(step_list), 4, 1))
            log_acs   = torch.zeros((len(step_list), 1))
        
            for index, i in enumerate(step_list):
                global_state[index] = torch.from_numpy(np.array(mem[i][0]))
                global_acs[index]   = mem[i][1]
                log_acs[index] = mem[i][-1]
            curr_q = value_net(global_state, global_acs)
            p_loss += - torch.sum(curr_q.clone().detach() * log_acs)
        p_loss /= 10
          
        optimizer2.zero_grad()
        p_loss.backward(retain_graph=True, inputs=list(policy_net.parameters()))
        grad_norm_p = torch.nn.utils.clip_grad_norm(
                policy_net.parameters(), 0.5)
        optimizer2.step()
        soft_update(policy_net_tar, policy_net, 0.03)

        logger.add_scalar('p_loss/loss', p_loss, Mem.niter_p)
        logger.add_scalar('p_loss/grad', grad_norm_p, Mem.niter_p)
        Mem.niter_p += 1
    print("base policy update successful!")
    
    #update perturbation policy
    ##long term cost
    for _ in range(1):
        loss = 0
        for ii in range(10):
            step_list = random.choices(range(len(mem)), k=int(len(mem) * 0.3))
            global_state = torch.zeros((len(step_list), 4, 5, 4))
            cost         = torch.zeros((len(step_list), 1))
            acc_cost     = torch.zeros((len(step_list), 1))
        
            for index, i in enumerate(step_list):
                global_state[index] = torch.from_numpy(np.array(mem[i][0]))
                cost = mem[i][3]
                acc_cost = mem[i][4]

            acs = sample_action(global_state, policy_net, cost_net, update=False)
            c_value = c1_net(global_state, acs)    
    
            bound = 20
            xxx1 = torch.clamp(c_value + acc_cost - cost - bound, 0)
            xxx1 = ((xxx1 - xxx1.mean()) / (1e-6 + xxx1.var()))**2
            loss += xxx1.mean()
            
        optimizer_cost.zero_grad()
        loss.backward(retain_graph=True, inputs=list(cost_net.parameters()))
        grad_norm_cost = torch.nn.utils.clip_grad_norm(
                cost_net.parameters(), 0.5)
        optimizer_cost.step()
        soft_update(cost_net_tar, cost_net, 0.01)
        
        logger.add_scalar('cost_policy_loss/loss', loss, Mem.niter_p_cost)
        logger.add_scalar('cost_policy_loss/grad', grad_norm_cost, Mem.niter_p_cost)
        Mem.niter_p_cost += 1
    print("perturbation policy update successful!")
    
    
        
        

