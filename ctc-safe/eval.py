import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.decom import DeCoM
import sys
from datetime import datetime as dt
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=False)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):

    run_num = config.seed
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)

    model = DeCoM.init_from_save(config.path)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    reward = []
    cost1 = []
    cost2 = []
    cost3 = []
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        t += 1

        acc_cost = np.array([[[0.0]*3]*model.nagents]*config.n_rollout_threads)
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            agent_actions_cp = copy.deepcopy(agent_actions)
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, costs, dones, infos = env.step(actions)

            acc_cost += costs
            #print(et_i, acc_cost)

            replay_buffer.push(obs, agent_actions_cp, rewards, costs, next_obs, dones, np.array([[1 if et_i==0 else 0]*model.nagents]*config.n_rollout_threads), acc_cost)
            obs = next_obs

        ep_rews, ep_costs = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        
        for a_i, a_ep_rew in enumerate(ep_rews):
            if a_i == 0:
                #agents share reward and costs
                reward.append(a_ep_rew * config.episode_length)
                cost1.append(ep_costs[a_i][0] * config.episode_length)
                cost2.append(ep_costs[a_i][1] * config.episode_length)
                cost3.append(ep_costs[a_i][2] * config.episode_length)
            
    print(np.mean(reward), np.std(reward))
    print(np.mean(cost1),  np.std(cost1))
    print(np.mean(cost2),  np.std(cost2))
    print(np.mean(cost3),  np.std(cost3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("--n_rollout_threads", default=24, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=8, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", default=False, type=bool)
    parser.add_argument("--path", help="path to models to load")
    parser.add_argument("--seed", default=0,type=int)
    config = parser.parse_args()

    run(config)
    
