"""
reward.py
This file describes reward function which is the “expected reward” for the belief distribution over Gaussian reward distribution.
rew_std: standard deviation of Gaussian distribution for reward [std_x, std_y]

b(s)= 1/sqrt(2*pi*det(P)) * exp(-0.5* ((s-x)^T*P^-1*(s-x)) : Gaussian distribution with mean x, covariance P
r(s) = scale * exp(-0.5* s^T* R^-1 * s): reward gaussian distribution with mean zeros, covariance R
invS = invR +invP
R(b) = \int b(s)*r(s) ds = c *sqrt(det(S)/det(P))* exp(-0.5* mu^T*(invP - invP*S*invP)*mu)

R(b) =  \int b(s)*r(s) ds = 1/sqrt(det(2 pi (P+R)) * exp(-0.5*mu^t(R+P)^-1*mu)
"""
import torch
import numpy as np
from FireflyEnv.env_utils import is_pos_def


def return_reward(episode, info, reached_target, b, goal_radius, REWARD, finetuning = 0):
    if info['stop']:  # receive reward if monkey stops. position does not matters
        if finetuning == 0:
            reward = get_reward(b, goal_radius, REWARD)
            if reached_target == 1:
                print("Ep {}: Good Job!!, reward= {:0.3f}".format(episode, reward[-1]))
            else:
                pass #print("reward= %.3f" % reward.view(-1)) #pass
        elif finetuning == 1 and reached_target == 1:
            reward = REWARD * torch.ones(1)
            print("Ep {}: Good Job!!, FIXED reward= {:0.3f}".format(episode, reward[-1]))
        else:  # finetuning == 1 and reached_target == 0
            reward = -0 * torch.ones(1)
    else:
        reward = -0 * torch.ones(1)
    return reward


def get_reward(b, goal_radus, REWARD):
    bx, P = b
    rew_std = goal_radus / 2  # std of reward function --> 2*std (=goal radius) = reward distribution

    #rew_std = goal_radus/2/2 #std of reward function --> 2*std (=goal radius) = reward distribution
    reward = rewardFunc(rew_std, bx.view(-1), P, REWARD)  # reward currently only depends on belief not action
    return reward

"""
def rewardFunc(rew_std, x, P, scale):
    R = torch.eye(2) * rew_std**2 # reward function is gaussian
    P = P[:2, :2] # cov
    invP = torch.inverse(P)
    invS = torch.inverse(R) + invP
    S = torch.inverse(invS)
    mu = x[:2] # pos
    alpha = -0.5 * mu.matmul(invP - invP.mm(S).mm(invP)).matmul(mu)
    reward = torch.exp(alpha) * torch.sqrt(torch.det(S)/torch.det(P))
    reward = scale * reward # adjustment for reward per timestep
    return reward.view(-1)
"""

def rewardFunc(rew_std, x, P, scale):
    mu = x[:2]  # pos
    R = torch.eye(2) * rew_std**2 # reward function is gaussian
    P = P[:2, :2] # cov
    S = R+P
    if not is_pos_def(S):
        print('R+P is not positive definite!')
    alpha = -0.5 * mu @ S.inverse() @ mu.t()
    #alpha = -0.5 * mu.matmul(torch.inverse(R+P)).matmul(mu.t())
    reward = torch.exp(alpha) /2 / np.pi /torch.sqrt(S.det())

    # normalization -> to make max reward as 1
    mu_zero = torch.zeros(1,2)
    alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
    reward_zero = torch.exp(alpha_zero) /2 / np.pi /torch.sqrt(R.det())
    reward = reward/reward_zero
    ####################

    reward = scale * reward  # adjustment for reward per timestep
    if reward > scale:
        print('reward is wrong!', reward)
        print('mu', mu)
        print('P', P)
        print('R', R)
    return reward.view(-1)