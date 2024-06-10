from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import ContPolicy

class Agent(object):
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.base_policy = ContPolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.base_target_policy = ContPolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.base_target_policy, self.base_policy)
        self.base_policy_optimizer = Adam(self.base_policy.parameters(), lr=lr)

        self.pert_policy = ContPolicy(num_in_pol + num_out_pol*4, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.pert_target_policy = ContPolicy(num_in_pol + num_out_pol*4,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.pert_target_policy, self.pert_policy)
        self.pert_policy_optimizer = Adam(self.pert_policy.parameters(), lr=lr*3)

    def step1(self, obs, explore=True, trgt=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if not trgt:
            return self.base_policy(obs, sample=explore)
        else:
            return self.base_target_policy(obs, sample=explore)

    def step2(self, obs, explore=True, trgt=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if not trgt:
            return self.pert_policy(obs, sample=explore)
        else:
            return self.pert_target_policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.base_policy.state_dict(),
                'target_policy': self.base_target_policy.state_dict(),
                'policy_optimizer': self.base_policy_optimizer.state_dict(),
                'cost_policy': self.pert_policy.state_dict(),
                'cost_target_policy': self.pert_target_policy.state_dict(),
                'cost_policy_optimizer': self.pert_policy_optimizer.state_dict()}

    def load_params(self, params):
        self.base_policy.load_state_dict(params['policy'])
        self.base_target_policy.load_state_dict(params['target_policy'])
        self.base_policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.pert_policy.load_state_dict(params['cost_policy'])
        self.pert_target_policy.load_state_dict(params['cost_target_policy'])
        self.pert_policy_optimizer.load_state_dict(params['cost_policy_optimizer'])
