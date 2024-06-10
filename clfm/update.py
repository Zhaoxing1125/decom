import torch
import random
import math
import numpy as np
from torch.autograd import Variable

from utils import *
import random


def update_alg(Mem, mem, value_net, optimizer1, policy_net, optimizer2, logger, value_net_tar, policy_net_tar, \
               c1_net, c2_net, c1_net_tar, c2_net_tar, optimizer3, optimizer4, cost_net, cost_net_tar, optimizer_cost):
    
    #update critics
    random.seed(3)
    for _ in range(20):
        id_list = random.choices(list(mem.keys()), k=10)
        
        q_loss = 0
        c1_loss = 0
        c2_loss = 0
        for id in id_list:
            state_len = len(mem[id][0][0])

            state   = torch.zeros((len(mem[id]), state_len, 3))

            obs     = torch.zeros((len(mem[id]), 21))
            nei_obs = torch.zeros((len(mem[id]), 6, 7, 3))
            acs     = torch.zeros((len(mem[id]), 3))
            avg_acs = torch.zeros((len(mem[id]), 3))
            reward  = torch.zeros((len(mem[id]), 1))
            kl      = torch.zeros((len(mem[id]), 1))
            var     = torch.zeros((len(mem[id]), 1))
            
            for i, item in enumerate(mem[id]):
                state[i]  = torch.tensor(item[0], dtype=torch.float)
                obs[i] = torch.tensor(item[-3]).reshape(-1, 21)
                
                nei_obs[i] = torch.tensor(item[-2]).reshape(-1, 6, 7, 3)
                acs[i] = item[2]
                
                avg_acs[i] = item[3]
                reward[i] = item[4]
                kl[i] = item[5]
                var[i] = item[6]
                
            acs = step(obs, nei_obs, policy_net, cost_net)
            curr_q = value_net(state[:-1], acs[:-1], avg_acs[:-1])
            
            curr_c1 = c1_net(state[:-1], acs[:-1], avg_acs[:-1])
            curr_c2 = c2_net(state[:-1], acs[:-1], avg_acs[:-1])

            acs = step(obs, nei_obs, policy_net_tar, cost_net_tar)        
            next_q = value_net_tar(state[1:],  acs[1:].clone().detach(),   avg_acs[1:]).detach()
            
            next_c1 = c1_net_tar(state[1:],  acs[1:].clone().detach(),   avg_acs[1:]).detach()
            next_c2 = c2_net_tar(state[1:],  acs[1:].clone().detach(),   avg_acs[1:]).detach()


                
            q_ind_loss  = 0
            c1_ind_loss = 0
            c2_ind_loss = 0
            prev_t = mem[id][0][-1]
            for i, cq, nq, cc1, nc1, cc2, nc2 in zip(range(len(mem[id])), curr_q, next_q, curr_c1, next_c1, curr_c2, next_c2):
                q_ind_loss += (cq - reward[i] - nq * 0.99 ** (mem[id][i][-1]-prev_t) )**2
                c1_ind_loss += (cc1 - kl[i] - nc1 * 1.0 ** (mem[id][i][-1]-prev_t) )**2
                c2_ind_loss += (cc2 - var[i] - nc2 * 1.0 ** (mem[id][i][-1]-prev_t) )**2
                prev_t = mem[id][i][-1]

            q_loss += q_ind_loss / (1+len(mem[id]))
            c1_loss += c1_ind_loss / (1+len(mem[id]))
            c2_loss += c2_ind_loss / (1+len(mem[id]))

        optimizer1.zero_grad()
        q_loss.backward(retain_graph=True, inputs=list(value_net.parameters()))
        grad_norm_q = torch.nn.utils.clip_grad_norm(
                value_net.parameters(), 0.5)
        optimizer1.step()
        soft_update(value_net_tar, value_net, 0.0025)
        
        optimizer3.zero_grad()
        c1_loss.backward(retain_graph=True, inputs=list(c1_net.parameters()))
        grad_norm_c1 = torch.nn.utils.clip_grad_norm(
                c1_net.parameters(), 0.5)
        optimizer3.step()
        soft_update(c1_net_tar, c1_net, 0.0025)
        
        optimizer4.zero_grad()
        c2_loss.backward(retain_graph=True, inputs=list(c2_net.parameters()))
        grad_norm_c2 = torch.nn.utils.clip_grad_norm(
                c2_net.parameters(), 0.5)
        optimizer4.step()
        soft_update(c2_net_tar, c2_net, 0.0025)

        logger.add_scalar('q_loss/loss', q_loss, Mem.niter_q)
        logger.add_scalar('q_loss/grad', grad_norm_q, Mem.niter_q)
        logger.add_scalar('c1_loss/loss', c1_loss, Mem.niter_q)
        logger.add_scalar('c1_loss/grad', grad_norm_c1, Mem.niter_q)
        logger.add_scalar('c2_loss/loss', c2_loss, Mem.niter_q)
        logger.add_scalar('c2_loss/grad', grad_norm_c2, Mem.niter_q)
        Mem.niter_q += 1
    
    
    #update base policy
    id_list = random.choices(list(mem.keys()), k=100)
    state_len = len(mem[id_list[0]][0][0])

    state   = torch.zeros((len(id_list), state_len, 3))
    obs     = torch.zeros((len(id_list), 21))
    nei_obs = torch.zeros((len(id_list), 6, 7, 3))
    acs     = torch.zeros((len(id_list), 3))
    avg_acs = torch.zeros((len(id_list), 3))
    
    for i, id in enumerate(id_list):
        item = mem[id][0]
        state[i]  = torch.tensor(item[0], dtype=torch.float)
        obs[i] = torch.tensor(item[-3]).reshape(-1, 21)
        nei_obs[i] = torch.tensor(item[-2]).reshape(-1, 6, 7, 3)
        acs[i] = item[2]
        avg_acs[i] = item[3]

    acs, log_acs = step(obs, nei_obs, policy_net, cost_net, update=True)
    xxx1 = value_net(state, acs, avg_acs)    
    
    x1 = log_acs * xxx1.clone().detach()
    x1 = x1.mean() + xxx1.mean()
    p_loss = -x1
    
    optimizer2.zero_grad()
    p_loss.backward(retain_graph=True, inputs=list(policy_net.parameters()))
    grad_norm_p = torch.nn.utils.clip_grad_norm(
            policy_net.parameters(), 0.5)
    optimizer2.step()
    soft_update(policy_net_tar, policy_net, 0.1)
    
    
    
    #update perturbation policy
    ##long term cost
    id_list = random.choices(list(mem.keys()), k=50)
    state_len = len(mem[id_list[0]][0][0])

    state   = torch.zeros((len(id_list), state_len, 3))
    obs     = torch.zeros((len(id_list), 21))
    nei_obs = torch.zeros((len(id_list), 6, 7, 3))
    acs     = torch.zeros((len(id_list), 3))
    avg_acs = torch.zeros((len(id_list), 3))
    kl      = torch.zeros((len(id_list), 1))
    var      = torch.zeros((len(id_list), 1))
    
    for i, id in enumerate(id_list):
        item = mem[id][0]
        state[i]  = torch.tensor(item[0], dtype=torch.float)
        obs[i] = torch.tensor(item[-3]).reshape(-1, 21)
        nei_obs[i] = torch.tensor(item[-2]).reshape(-1, 6, 7, 3)
        acs[i] = item[2]
        avg_acs[i] = item[3]
        kl[i] = item[5]
        var[i] = item[6]

    acs = step(obs, nei_obs, policy_net, cost_net)
    xxx1 = c1_net(state, acs, avg_acs)    
    xxx2 = c2_net(state, acs, avg_acs)
    
    bound = [90.0, 60.0]
    
    xxx1 = torch.clamp(xxx1-bound[0], 0)
    xxx2 = torch.clamp(xxx2-bound[1], 0)
    
    xx1 = ((xxx1 - xxx1.mean()) / (1e-6 + xxx1.var()))**2
    xx2 = ((xxx2 - xxx2.mean()) / (1e-6 + xxx2.var()))**2
    
    loss_we = torch.max(xx1.mean(), xx2.mean())
    
    ##cost at each state
    id_list = random.choices(list(mem.keys()), k=100)
    loss_es = 0
    for id in id_list:
        state_len = len(mem[id][0][0])

        state   = torch.zeros((len(mem[id]), state_len, 3))

        obs     = torch.zeros((len(mem[id]), 21))
        nei_obs = torch.zeros((len(mem[id]), 6, 7, 3))
        acs     = torch.zeros((len(mem[id]), 3))
        avg_acs = torch.zeros((len(mem[id]), 3))
        
        kl      = torch.zeros((len(mem[id]), 1))
        var     = torch.zeros((len(mem[id]), 1))
        
        acc_kl  = torch.zeros((len(mem[id]), 1))
        acc_var = torch.zeros((len(mem[id]), 1))
        
        for i, item in enumerate(mem[id]):
            state[i]  = torch.tensor(item[0], dtype=torch.float)
            obs[i] = torch.tensor(item[-3]).reshape(-1, 21)
            
            nei_obs[i] = torch.tensor(item[-2]).reshape(-1, 6, 7, 3)
            acs[i] = item[2]
            
            avg_acs[i] = item[3]
            
            kl[i] = item[5]
            var[i] = item[6]
            
            acc_kl[i] = item[7][0]
            acc_var[i] = item[7][1]
            
        acs = step(obs, nei_obs, policy_net, cost_net)
        xxx1 = c1_net(state, acs, avg_acs)    
        xxx2 = c2_net(state, acs, avg_acs)
        
        bound = [90.0, 60.0]
        xxx1 = torch.clamp(xxx1 + acc_kl - kl - bound[0], 0)
        xxx2 = torch.clamp(xxx2 + acc_var - var - bound[1], 0)
        
        xx1 = ((xxx1 - xxx1.mean()) / (1e-6 + xxx1.var()))**2
        xx2 = ((xxx2 - xxx2.mean()) / (1e-6 + xxx2.var()))**2
        
        loss_es += torch.max(xx1.mean(), xx2.mean())
        
    loss_oa = torch.max(loss_we, loss_es/len(id_list))
    
    optimizer_cost.zero_grad()
    loss_oa.backward(retain_graph=True, inputs=list(cost_net.parameters()))
    grad_norm_cost = torch.nn.utils.clip_grad_norm(
            cost_net.parameters(), 0.5)
    optimizer_cost.step()
    soft_update(cost_net_tar, cost_net, 0.03)
    
    logger.add_scalar('p_loss/loss', p_loss, Mem.niter_p)
    logger.add_scalar('p_loss/grad', grad_norm_p, Mem.niter_p)
    logger.add_scalar('cost/loss', loss_oa, Mem.niter_p)
    logger.add_scalar('cost/grad', grad_norm_cost, Mem.niter_p)
    Mem.niter_p += 1
        
        

