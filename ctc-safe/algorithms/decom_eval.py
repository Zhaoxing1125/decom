import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import Agent
from utils.critics import Critic, cost_Critic
import sys
import copy

MSELoss = torch.nn.MSELoss()

class DeCoM(object):
    """
    Decomposed policy for constrained marl
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.99, tau=0.01, pi_lr=0.01, q_lr=0.01,
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

        self.cost_critic = cost_Critic(sa_size, hidden_dim=critic_hidden_dim)
        self.cost_target_critic = cost_Critic(sa_size, hidden_dim=critic_hidden_dim)
        hard_update(self.cost_target_critic, self.cost_critic)
        self.cost_critic_optimizer = Adam(self.cost_critic.parameters(), lr=q_lr*3,
                                     weight_decay=1e-3)



        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'


    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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
        #perturbation policy
        a = [policy.step2(torch.cat((obs, x1), dim=1), explore=explore, trgt=trgt) for policy, obs in zip(self.agents, observations)]
        
        #combine
        ac = [torch.clamp(b1+a1, -1, 1) for b1, a1 in zip(b, a)]

        if return_a:
            return ac, a
        else:
            return ac
            
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
                                       'cost_critic': self.cost_critic.state_dict(),
                                       'cost_target_critic': self.cost_target_critic.state_dict(),
                                       'cost_critic_optimizer': self.cost_critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01, w_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128,
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
            instance.cost_critic.load_state_dict(critic_params['cost_critic'])
            instance.cost_target_critic.load_state_dict(critic_params['cost_target_critic'])
            instance.cost_critic_optimizer.load_state_dict(critic_params['cost_critic_optimizer'])
        return instance
