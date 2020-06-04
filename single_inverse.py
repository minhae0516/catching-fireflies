import torch
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import grad
from InverseFuncs import getLoss, reset_theta, theta_range

from collections import deque

import torch
import numpy as np
import time

import matplotlib.pyplot as plt



def single_inverse(true_theta, arg, env, agent, x_traj, a_traj,  true_loss, filename, n, Pro_Noise = None, Obs_Noise = None, part_theta=False):
    tic = time.time()

    if Pro_Noise is not None:
        Pro_Noise = true_theta[2:4]
    if Obs_Noise is not None:
        Obs_Noise = true_theta[6:8]

    #rndsgn = torch.sign(torch.randn(1,len(true_theta))).view(-1)
    #purt= torch.Tensor([0.1,0.1,1,1,0.1,0.1,1,1,0.1])#perturbation

    #theta = nn.Parameter(true_theta.data.clone()+purt *torch.randn(1,len(true_theta)).view(-1))
    #theta = theta_range(theta, arg.gains_range, arg.noise_range, arg.goal_radius_range)  # keep inside of trained range


    # just for checking
    #theta = nn.Parameter(true_theta.data.clone())  # just for checking


    theta = nn.Parameter(reset_theta(arg.gains_range, arg.noise_range, arg.goal_radius_range))
    ini_theta = theta.data.clone()


    loss_log = deque(maxlen=arg.NUM_IT)
    loss_log_recent = deque(maxlen=100)
    #loss_act_log = deque(maxlen=arg.NUM_IT)
    #loss_obs_log = deque(maxlen=arg.NUM_IT)
    theta_log = deque(maxlen=arg.NUM_IT)

    optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)


    for it in tqdm(range(arg.NUM_IT)):
        loss, loss_act, loss_obs = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.noise_range, arg.PI_STD, arg.NUM_SAMPLES)
        loss_log.append(loss.data)
        loss_log_5.append(loss.data)
        #loss_act_log.append(loss_act.data)
        #loss_obs_log.append(loss_obs.data)
        optT.zero_grad() #clears old gradients from the last step
        loss.backward(retain_graph=True) #computes the derivative of the loss w.r.t. the parameters using backpropagation
        optT.step() # performing single optimize step: this changes theta
        if part_theta == False:
            theta = theta_range(theta, arg.gains_range, arg.noise_range, arg.goal_radius_range) # keep inside of trained range
        elif part_theta == True:
            theta = theta_range(theta, arg.gains_range, arg.noise_range,
                                arg.goal_radius_range, Pro_Noise, Obs_Noise)  # keep inside of trained range


        theta_log.append(theta.data.clone())

        if it%5 == 0:
            #print("num_theta:{}, num:{}, loss:{}".format(n, it, np.round(loss.data.item(), 6)))
            #print("num:{},theta diff sum:{}".format(it, 1e6 * (true_theta - theta.data.clone()).sum().data))
            print("num_theta:{}, num:{}, loss:{}, true loss:{},\n true_theta:{}, \n converged_theta:{}\n".format(n, it,np.round(loss.data.item(), 6),np.round(true_loss.data.item(), 6),true_theta.data, theta.data))

        if it%50 == 0 and it >0:
            plt.plot(loss_log)
            plt.title("it:{}".format(it))
            plt.savefig('../firefly-inverse-data/data/'+filename +str(n)+'_loss.png')


        if it >200 and it%10==0:
            if np.mean(loss_log_recent) < true_loss:
                break


    toc = time.time()
    # print((toc - tic)/60/60, "hours")
    """
    loss, _, _ = getLoss(agent, x_traj, a_traj, theta, env, arg.gains_range, arg.noise_range, arg.PI_STD, arg.NUM_SAMPLES)
    #print("loss:{}".format(loss))

    grads = grad(loss, theta, create_graph=True)[0]
    H = torch.zeros(9,9)
    for i in range(9):
        H[i] = grad(grads[i], theta, retain_graph=True)[0]
    I = H.inverse()
    stderr = torch.sqrt(torch.abs(I).diag())


    stderr_ii = 1/torch.sqrt(torch.abs(H.diag()))
    """



    result = {'true_theta': true_theta,
              'initial_theta': ini_theta,
              'x_traj': x_traj,
              'a_traj': a_traj,
              'theta': theta,
              'theta_log': theta_log,
              'loss_log': loss_log,
              'true_loss': true_loss,
              'filename': filename,
              'num_theta': n,
              'converging_it': it,
              'duration': toc-tic,
              'arguments': arg}
              #'stderr': stderr,
              #'stderr_ii': stderr_ii
              #}
            #'loss_act_log': loss_act_log,
            #'loss_obs_log': loss_obs_log,

    torch.save(result, '../firefly-inverse-data/data/' + filename + str(n)+ str(arg.NUM_thetas) + "EP" + str(arg.NUM_EP) + str(np.around(arg.PI_STD, decimals=2)) + str(arg.NUM_SAMPLES) + "IT" + str(arg.NUM_IT) + str(arg.SEED_NUMBER) +'_single_result_part.pkl')

    return result