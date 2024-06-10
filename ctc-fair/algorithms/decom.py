import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import Agent
from utils.critics import Critic, cost_Critic
import sys
import copy
import numpy as np

MSELoss = torch.nn.MSELoss()

class DeCoM(object):
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)

        self.agents = [Agent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        self.critic = Critic(sa_size, hidden_dim=critic_hidden_dim)
        self.target_critic = Critic(sa_size, hidden_dim=critic_hidden_dim)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)


        #fairness
        self.cost_critic_fairness = cost_Critic(sa_size, hidden_dim=critic_hidden_dim)
        self.cost_target_critic_fairness = cost_Critic(sa_size, hidden_dim=critic_hidden_dim)
        hard_update(self.cost_target_critic_fairness, self.cost_critic_fairness)
        self.cost_critic_optimizer_fairness = Adam(self.cost_critic_fairness.parameters(), lr=q_lr*3,
                                    weight_decay=1e-3)

        #stability
        self.cost_critic_stability = cost_Critic(sa_size, hidden_dim=critic_hidden_dim)
        self.cost_target_critic_stability = cost_Critic(sa_size, hidden_dim=critic_hidden_dim)
        hard_update(self.cost_target_critic_stability, self.cost_critic_stability)
        self.cost_critic_optimizer_stability = Adam(self.cost_critic_stability.parameters(), lr=q_lr*3,weight_decay=1e-3)



        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.cost_trgt_critic_fairness_dev = 'cpu' 
        self.cost_trgt_critic_stability_dev = 'cpu' 
        self.cost_critic_fairness_dev = 'cpu' 
        self.cost_critic_stability_dev = 'cpu' 
        self.niter = 0
        self.cost_niter = 0


    def step(self, observations, explore=True, trgt=False, return_a=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        #base policy
        b = [policy.step1(obs, explore=explore, trgt=trgt) for policy, obs in zip(self.agents, observations)]
        x1 = torch.stack(b, dim=1).reshape(-1, 8)
        #perturbation
        a = [policy.step2(obs, x1, explore=explore, trgt=trgt) for policy, obs in zip(self.agents, observations)]
        #combine
        ac = [torch.clamp(b1+a1, -1, 1) for b1, a1 in zip(b, a)]

        if return_a:
            return ac, a
        else:
            return ac

    def update_pert_policy(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, costs, acc_costs, next_obs, dones, steps = sample
        
        samp_acs, samp_a = self.step(obs, trgt=False, return_a=True, explore=False)
        critic_in = list(zip(obs, samp_acs))
        
        c_critic_rets = [self.cost_critic_fairness(critic_in),
                        self.cost_critic_stability(critic_in)]

        bound = [0, 60]


        for a_i in range(self.nagents):
            curr_agent = self.agents[a_i]


            backw_cost1 = acc_costs[a_i][:, 0].reshape(len(obs[0]), 1)
            forwa_cost1 = c_critic_rets[0][a_i].reshape(len(obs[0]), 1)
            imme_cost1  = costs[a_i][:, 0].reshape(len(obs[0]), 1)
            
            cost1_every_state_cost = torch.clamp(forwa_cost1+backw_cost1-imme_cost1-bound[0], min=0)
            
            cost1_every_state_cost_mean = cost1_every_state_cost.mean().clone().detach()
            cost1_every_state_cost_std  = cost1_every_state_cost.std().clone().detach() + 1e-6
            #print(cost1_every_state_cost, cost1_every_state_cost_mean, cost1_every_state_cost_std)
            cost1_final_cost = torch.clamp(forwa_cost1 * steps[a_i].view(-1, 1)-bound[0], min=0)
            
            cost1_final_cost_mean = cost1_final_cost.mean().clone().detach()
            cost1_final_cost_std  = cost1_final_cost.std().clone().detach() + 1e-6
            #print(cost1_final_cost, cost1_final_cost_mean, cost1_final_cost_std) 

            xxx1 = torch.max(torch.square((cost1_every_state_cost-cost1_every_state_cost_mean)/cost1_every_state_cost_std).mean(), \
                            torch.square((cost1_final_cost-cost1_final_cost_mean)/cost1_final_cost_std).mean())

            #print(cost1_every_state_cost_mean, cost1_every_state_cost_std)
            #print("-----------------------")
            #print(cost1_final_cost_mean, cost1_final_cost_std)
            #print("-----------------------")
            #print(xxx1)

            dominant_index = 0 if ((cost1_every_state_cost-cost1_every_state_cost_mean)/cost1_every_state_cost_std).mean() > ((cost1_final_cost-cost1_final_cost_mean)/cost1_final_cost_std).mean() else 1


            
            #backw_cost2 = acc_costs[a_i][:, 1].reshape(len(obs[0]), 1)
            #forwa_cost2 = c_critic_rets[1][a_i].reshape(len(obs[0]), 1)
            #imme_cost2  = costs[a_i][:, 1].reshape(len(obs[0]), 1)

            #cost2_every_state_cost = torch.clamp(forwa_cost2+backw_cost2-imme_cost2-bound[1], min=0)
            #cost2_every_state_cost_mean = cost2_every_state_cost.mean().clone().detach()
            #cost2_every_state_cost_std  = cost2_every_state_cost.std().clone().detach()+ 1e-6

            #cost2_final_cost = torch.clamp(forwa_cost2 * steps[a_i].view(-1, 1)-bound[1], min=0)
            
            #cost2_final_cost_mean = cost2_final_cost.mean().clone().detach()
            #cost2_final_cost_std  = cost2_final_cost.std().clone().detach()+ 1e-6

            #xxx2 = torch.max(torch.square((cost2_every_state_cost-cost2_every_state_cost_mean)/cost2_every_state_cost_std).mean(), \
            #                torch.square((cost2_final_cost-cost2_final_cost_mean)/cost2_final_cost_std).mean())

            #print(cost2_every_state_cost_mean, cost2_every_state_cost_std)
            #print("-----------------------")
            #print(cost2_final_cost_mean, cost2_final_cost_std)
            #print("-----------------------")
            #print(xxx2)

            #if xxx2 > xxx1:
            #    dominant_index = 2 if ((cost2_every_state_cost-cost2_every_state_cost_mean)/cost2_every_state_cost_std).mean() > ((cost2_final_cost-cost2_final_cost_mean)/cost2_final_cost_std).mean() else 3



            
            pol_loss = xxx1 #torch.max(xxx1, xxx2) #+ 0.01*(samp_a[a_i].mean())**2  
            #print(pol_loss)
            #print("----------") 
            #sys.exit(0)   

            pol_loss.backward(retain_graph=True, inputs=list(curr_agent.cost_policy.parameters()))
            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.cost_policy.parameters(), 0.5)

            
            
            curr_agent.cost_policy_optimizer.step()
            curr_agent.cost_policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/cost_pol_loss' % a_i,
                                pol_loss, self.cost_niter)
                logger.add_scalar('agent%i/grad_norms/cost_pi' % a_i,
                                grad_norm, self.cost_niter)
                logger.add_scalar('agent%i/dominant/index' % a_i,
                                dominant_index, self.cost_niter)
        
        self.cost_niter += 1

    def update_cost_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for costs
        """
        obs, acs, rews, costs, _, next_obs, dones, steps = sample
        #input
        curr_next_ac = self.step(next_obs, trgt=True)
        trgt_critic_in = list(zip(next_obs, curr_next_ac))
        critic_in = list(zip(obs, acs))
        
        #Q
        c_next_qs1 = self.cost_target_critic_fairness(trgt_critic_in)
        c_critic_rets1 = self.cost_critic_fairness(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        c_next_qs2 = self.cost_target_critic_stability(trgt_critic_in)
        c_critic_rets2 = self.cost_critic_stability(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)

        c_q_loss = 0
        c_q_loss_stability = 0
        for a_i, c_nq, c_pq in zip(range(self.nagents), c_next_qs1, c_critic_rets1):
            c_target_q = (costs[a_i][:, 0].view(-1, 1) +
                        1.0 * c_nq *
                        (1 - dones[a_i].view(-1, 1)))

            c_q_loss += MSELoss(c_pq, c_target_q.detach())

        for a_i, c_nq, c_pq in zip(range(self.nagents), c_next_qs2, c_critic_rets2):
            c_target_q = (costs[a_i][:, 1].view(-1, 1) +
                        1.0 * c_nq *
                        (1 - dones[a_i].view(-1, 1)))

            c_q_loss_stability += MSELoss(c_pq, c_target_q.detach())

        #fairness
        self.cost_critic_optimizer_fairness.zero_grad()
        c_q_loss.backward()
        
        c_grad_norm = torch.nn.utils.clip_grad_norm(
            self.cost_critic_fairness.parameters(), 10 * self.nagents)
        self.cost_critic_optimizer_fairness.step()

        #stability
        self.cost_critic_optimizer_stability.zero_grad()
        c_q_loss_stability.backward()
        
        c_grad_norm_stability = torch.nn.utils.clip_grad_norm(
            self.cost_critic_stability.parameters(), 10 * self.nagents)
        self.cost_critic_optimizer_stability.step()

        if logger is not None:
            logger.add_scalar('cost1/cq_loss', c_q_loss, self.niter)
            logger.add_scalar('cost1/grad', c_grad_norm, self.niter)
            logger.add_scalar('cost2/cq_loss', c_q_loss_stability, self.niter)
            logger.add_scalar('cost2/grad', c_grad_norm_stability, self.niter)
        self.niter += 1


    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, costs, _, next_obs, dones, steps = sample
        # Q loss
        curr_next_ac = self.step(next_obs, trgt=True)
        trgt_critic_in = list(zip(next_obs, curr_next_ac))
        critic_in = list(zip(obs, acs))

        #reward
        r_next_qs = self.target_critic(trgt_critic_in)
        r_critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)

        r_q_loss = 0
        for a_i, r_nq, r_pq in zip(range(self.nagents), r_next_qs,
                                                r_critic_rets):
            r_target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * r_nq *
                        (1 - dones[a_i].view(-1, 1)))

            r_q_loss += MSELoss(r_pq, r_target_q.detach())

        self.critic_optimizer.zero_grad()
        r_q_loss.backward()
        
        r_grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()

        if logger is not None:
            logger.add_scalar('losses/rq_loss', r_q_loss, self.niter)
            logger.add_scalar('grad_norms/rq', r_grad_norm, self.niter)
        self.niter += 1

    def update_base_policies(self, sample, eps, soft=True, logger=None, **kwargs):
        obs, acs, rews, _, _, next_obs, dones, _ = sample

        samp_acs = self.step(obs, trgt=False, explore=False)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in)
        
        pol_loss = 0
        for a_i, q in zip(range(self.nagents), critic_rets):
            pol_loss += q.mean()
        pol_loss = -pol_loss

        pol_loss.backward(retain_graph=True)
        for a_j in range(self.nagents):
            curr_agent = self.agents[a_j]
            
            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.base_policy.parameters(), 0.5)

            curr_agent.base_policy_optimizer.step()
            curr_agent.base_policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_j,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_j,
                                  grad_norm, self.niter)
                                  

    def update_all_targets(self, eps):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        times = max(1, 5 - int(eps / 5000 ) * 0.5) 

        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.cost_target_critic_fairness, self.cost_critic_fairness, self.tau*times)
        soft_update(self.cost_target_critic_stability, self.cost_critic_stability, self.tau*times)
        for a in self.agents:
            soft_update(a.base_target_policy, a.base_policy, self.tau)
            soft_update(a.pert_target_policy, a.pert_policy, self.tau*times)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.cost_critic_fairness.train()
        self.cost_target_critic_fairness.train()
        self.cost_critic_stability.train()
        self.cost_target_critic_stability.train()
        for a in self.agents:
            a.base_policy.train()
            a.base_target_policy.train()
            a.pert_policy.train()
            a.pert_target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda(device='cuda:1')
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.base_policy = fn(a.base_policy)
                a.pert_policy = fn(a.pert_policy)
            self.pol_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.base_target_policy = fn(a.base_target_policy)
                a.pert_target_policy = fn(a.pert_target_policy)
            self.trgt_pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
        if not self.cost_critic_fairness_dev == device:
            self.cost_critic_fairness = fn(self.cost_critic_fairness)
            self.cost_critic_fairness_dev = device
        if not self.cost_trgt_critic_fairness_dev == device:
            self.cost_target_critic_fairness = fn(self.cost_target_critic_fairness)
            self.cost_trgt_critic_fairness_dev = device
        if not self.cost_critic_stability_dev == device:
            self.cost_critic_stability = fn(self.cost_critic_stability)
            self.cost_critic_stability_dev = device
        if not self.cost_trgt_critic_stability_dev == device:
            self.cost_target_critic_stability = fn(self.cost_target_critic_stability)
            self.cost_trgt_critic_stability_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.base_policy.eval()
            a.pert_policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.base_policy = fn(a.base_policy)
                a.pert_policy = fn(a.pert_policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict(),
                                       'cost_critic': self.cost_critic_fairness.state_dict(),
                                       'cost_target_critic': self.cost_target_critic_fairness.state_dict(),
                                       'cost_critic_optimizer': self.cost_critic_optimizer_fairness.state_dict(),
                                       'cost_critic_stability': self.cost_critic_stability.state_dict(),
                                       'cost_target_critic_stability': self.cost_target_critic_stability.state_dict(),
                                       'cost_critic_optimizer_stability': self.cost_critic_optimizer_stability.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01, w_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],
                                      'num_out_pol': acsp.shape[0]})
            sa_size.append((obsp.shape[0], acsp.shape[0]))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr, 'w_lr': w_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            instance.cost_critic_fairness.load_state_dict(critic_params['cost_critic'])
            instance.cost_target_critic_fairness.load_state_dict(critic_params['cost_target_critic'])
            instance.cost_critic_optimizer_fairness.load_state_dict(critic_params['cost_critic_optimizer'])
            instance.cost_critic_stability.load_state_dict(critic_params['cost_critic_stability'])
            instance.cost_target_critic_stability.load_state_dict(critic_params['cost_target_critic_stability'])
            instance.cost_critic_optimizer_stability.load_state_dict(critic_params['cost_critic_optimizer_stability'])
        return instance
