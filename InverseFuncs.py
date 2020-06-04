
# collections of functions for inverse control

import torch
import numpy as np
from numpy import pi
from FireflyEnv.env_utils import sample_exp

def trajectory(agent, theta, env, arg, gains_range, noise_range, goal_radius_range, NUM_EP):
    pro_gains, pro_noise_ln_vars, obs_gains, obs_noise_ln_vars,  goal_radius = torch.split(theta.view(-1), 2)

    x_traj = [] # true location
    obs_traj =[] # observation
    a_traj = [] # action
    b_traj = []
    x, _, _, _ = env.reset(gains_range, noise_range, goal_radius_range, goal_radius, pro_gains, pro_noise_ln_vars)
    #ox = agent.Bstep.observations(x)  # observation


    b, state, obs_gains, obs_noise_ln_vars = agent.Bstep.reset(x, torch.zeros(1), pro_gains, pro_noise_ln_vars, goal_radius, gains_range, noise_range, obs_gains, obs_noise_ln_vars)  # reset monkey's internal model
    episode = 0
    tot_t = 0

    #while tot_t <= TOT_T:
    while episode <= NUM_EP:
        episode +=1
        t = torch.zeros(1)
        x_traj_ep = []
        obs_traj_ep = []
        a_traj_ep = []
        b_traj_ep = []

        while t < arg.EPISODE_LEN: # for a single FF

            action = agent.actor(state)

            next_x, reached_target = env(x, action.view(-1)) #track true next_x of monkey
            next_ox = agent.Bstep.observations(next_x)  # observation
            #next_ox = agent.Bstep.observations_mean(next_x)  # observation no noise

            next_b, info = agent.Bstep(b, next_ox, action, env.box) # belief next state, info['stop']=terminal # reward only depends on belief
            next_state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_ln_vars, obs_gains, obs_noise_ln_vars, goal_radius)) # state used in policy is different from belief

            # check time limit
            TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
            mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

            x_traj_ep.append(x)
            obs_traj_ep.append(next_ox)
            a_traj_ep.append(action)
            b_traj_ep.append(b)

            x = next_x
            state = next_state
            b = next_b
            #ox = next_ox
            tot_t += 1.
            t += 1

            if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
                x, _, _, _ = env.reset(gains_range, noise_range, goal_radius_range, goal_radius, pro_gains, pro_noise_ln_vars)
                #ox = agent.Bstep.observations(x)  # observation
                b, state, _, _ = agent.Bstep.reset(x, torch.zeros(1), pro_gains, pro_noise_ln_vars, goal_radius, gains_range, noise_range, obs_gains, obs_noise_ln_vars)  # reset monkey's internal model
                break
        x_traj.append(x_traj_ep)
        obs_traj.append(obs_traj_ep)
        a_traj.append(a_traj_ep)
        b_traj.append(b_traj_ep)
    return x_traj, obs_traj, a_traj, b_traj



# MCEM based approach
def getLoss(agent, x_traj, a_traj, theta, env, gains_range, noise_range, PI_STD, NUM_SAMPLES):

    logPr = torch.zeros(1) #torch.FloatTensor([])
    logPr_act = torch.zeros(1)
    logPr_obs = torch.zeros(1)

    pro_gains, pro_noise_ln_vars, obs_gains, obs_noise_ln_vars, goal_radius = torch.split(theta.view(-1), 2)

    env.pro_gains = pro_gains
    env.pro_noise_ln_vars = pro_noise_ln_vars
    env.goal_radius = goal_radius


    for num_it in range(NUM_SAMPLES):

        for ep, x_traj_ep in enumerate(x_traj):
            a_traj_ep = a_traj[ep]
            logPr_ep = torch.zeros(1)
            logPr_act_ep = torch.zeros(1)
            logPr_obs_ep = torch.zeros(1)

            b_traj_log = []
            obs_traj_log = []

            t = torch.zeros(1)
            x = x_traj_ep[0].view(-1)
            b, state, _, _ = agent.Bstep.reset(x, t, pro_gains, pro_noise_ln_vars, goal_radius, gains_range,
                                               noise_range, obs_gains, obs_noise_ln_vars)  # reset monkey's internal model
            b_traj_log.append(b)

            for it, next_x in enumerate(x_traj_ep[1:]):
                action = agent.actor(state) # simulated acton



                next_ox_mean = agent.Bstep.observations_mean(next_x) # multiplied by observation gain, no noise
                #next_ox = next_ox_mean
                next_ox = agent.Bstep.observations(next_x)  # simulated observation (with noise)



                action_loss =5*torch.ones(2)+np.log(np.sqrt(2* pi)*PI_STD) + (action - a_traj_ep[it] ) ** 2 / 2 /(PI_STD**2)
                obs_loss = 5*torch.ones(2)+torch.log(np.sqrt(2* pi)*torch.sqrt(torch.exp(obs_noise_ln_vars))) +(next_ox - next_ox_mean).view(-1) ** 2/2/torch.exp(obs_noise_ln_vars)

                logPr_act_ep = logPr_act_ep + action_loss.sum()
                logPr_obs_ep = logPr_obs_ep + obs_loss.sum()



                logPr_ep = logPr_ep + logPr_act_ep + logPr_obs_ep
                #logPr_ep = logPr_ep +   (obs_loss).sum()

                next_b, info = agent.Bstep(b, next_ox, a_traj_ep[it], env.box)  # action: use real data
                next_state = agent.Bstep.Breshape(next_b, t, (pro_gains, pro_noise_ln_vars, obs_gains, obs_noise_ln_vars,
                                                              goal_radius))  # state used in policy is different from belief
                t += 1
                state = next_state
                b = next_b
                b_traj_log.append(b)
                obs_traj_log.append(next_ox)


            logPr_act += logPr_act_ep
            logPr_obs += logPr_obs_ep

            logPr += logPr_ep
            #logPr = torch.cat([logPr, logPr_ep])
            #print("b:\n", b_traj_log)
            #print("obs:\n",  obs_traj_log)

    return logPr/NUM_SAMPLES, logPr_act/NUM_SAMPLES, logPr_obs/NUM_SAMPLES #logPr.sum()




def reset_theta(gains_range, noise_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
    pro_gains = torch.zeros(2)
    obs_gains = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [proc_gain_ang]
    obs_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [obs_gain_ang]
    goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], goal_radius_range[1])


    if Pro_Noise is None:
        pro_noise_ln_vars= torch.zeros(2)
        # this uniform sampling returns low noise more frequently
        pro_noise_ln_vars[0] = torch.zeros(1).uniform_(noise_range[0], noise_range[1])# [proc_vel_noise]
        pro_noise_ln_vars[1] = torch.zeros(1).uniform_(noise_range[2], noise_range[3]) # [proc_ang_noise]

        # reason why to use exponential function: when the agent gets traiend, try use high noise => use it for training purpose
        #pro_noise_ln_vars[0] = -1 * sample_exp(-noise_range[1], -noise_range[0])  # [obs_vel_noise]
        #pro_noise_ln_vars[1] = -1 * sample_exp(-noise_range[3], -noise_range[2])  # [obs_ang_noise]
    else:
        pro_noise_ln_vars = Pro_Noise


    if Obs_Noise is None:
        obs_noise_ln_vars= torch.zeros(2)

        obs_noise_ln_vars[0] = torch.zeros(1).uniform_(noise_range[0], noise_range[1])  # [obs_vel_noise]
        obs_noise_ln_vars[1] = torch.zeros(1).uniform_(noise_range[2], noise_range[3]) # [obs_ang_noise]


        #obs_noise_ln_vars[0] = -1 * sample_exp(-noise_range[1], -noise_range[0])  # [obs_vel_noise]
        #obs_noise_ln_vars[1] = -1 * sample_exp(-noise_range[3], -noise_range[2])  # [obs_ang_noise]
    else:
        obs_noise_ln_vars = Obs_Noise

    theta = torch.cat([pro_gains, pro_noise_ln_vars, obs_gains, obs_noise_ln_vars, goal_radius])
    return theta


def theta_range(theta, gains_range, noise_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):

    theta[0].data.clamp_(gains_range[0], gains_range[1])
    theta[1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

    if Pro_Noise is None:
        theta[2].data.clamp_(noise_range[0], noise_range[1])  # [proc_vel_noise]
        theta[3].data.clamp_(noise_range[2], noise_range[3])  # [proc_ang_noise]
    else:
        theta[2:4].data.copy_(Pro_Noise.data)

    theta[4].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    theta[5].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

    if Obs_Noise is None:
        theta[6].data.clamp_(noise_range[0], noise_range[1])  # [obs_vel_noise]
        theta[7].data.clamp_(noise_range[2], noise_range[3])  # [obs_ang_noise]
    else:
        theta[6:8].data.copy_(Obs_Noise.data)

    theta[8].data.clamp_(goal_radius_range[0], goal_radius_range[1])

    # give some error if it hits the extreme
    theta_copy = theta.data.clone()
    for i in [0,4]:
        if theta[i] == gains_range[0]:
            theta[i].data.copy_(theta_copy[i] + 1e-2 * torch.rand(1).item())

        if theta[i] == gains_range[1]:
            theta[i].data.copy_(theta_copy[i] - 1e-2 * torch.rand(1).item())
    for i in [1, 5]:
        if theta[i] == gains_range[2]:
            theta[i].data.copy_(theta_copy[i] + 1e-2 * torch.rand(1).item())

        if theta[i] == gains_range[3]:
            theta[i].data.copy_(theta_copy[i] - 1e-2 * torch.rand(1).item())

    if theta[8] == goal_radius_range[0]:
        theta[8].data.copy_(theta_copy[8] + 1e-2 * torch.rand(1).item())
    elif theta[8] == goal_radius_range[1]:
        theta[8].data.copy_(theta_copy[8] - 1e-2 * torch.rand(1).item())
    del theta_copy
    return theta


def theta_init(agent, env, arg):
    # true theta
    true_theta = reset_theta(arg.gains_range, arg.noise_range, arg.goal_radius_range)
    #true_theta = torch.Tensor([1, np.pi / 4, -4, -4, 1, np.pi / 4, -4, -4, 0.4])
    #true_theta_log.append(true_theta.data.clone())
    x_traj, obs_traj, a_traj, b_traj = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.noise_range,
                                      arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory

    true_loss, true_loss_act, true_loss_obs = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.noise_range, arg.PI_STD,
                        arg.NUM_SAMPLES)  # this is the lower bound of loss?

    init_result = {'true_theta_log': true_theta,
                   'true_loss_log': true_loss,
                   'true_loss_act_log': true_loss_act,
                   'true_loss_obs_log': true_loss_obs,
                   'x_traj_log': x_traj,
                   'obs_traj_log': obs_traj,
                   'a_traj_log': a_traj,
                   'b_traj_log': b_traj
                   }
    return init_result
