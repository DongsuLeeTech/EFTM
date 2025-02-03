import torch
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import grad, Variable
from Inverse_coef import reset_coef, coef_bound
from Inverse_loss import Lossfn
from Copy_Actor_Net import *

from collections import deque

import numpy as np
import time

import matplotlib.pyplot as plt

def exp_inverse(true_coef, arg, env, agent, x_traj, a_traj, true_loss, filename, n):
    # start time
    tic = time.time()

    # copy parameter of actor
    w1, w2, w3, b1, b2, b3 = copy_params(agent)

    # random and pertubation
    rndco = torch.sign(torch.randn(1, len(true_coef))).view(-1)
    prt = torch.Tensor([-0.2, 0.3, -0.3])

    # Parameterization of coefficient to apply optimizer
    coef = Variable(true_coef - rndco * prt, requires_grad=True)
    coefp = Variable(true_coef - rndco * prt, requires_grad=False)
    coef = nn.Parameter(coef)

    # Bound coef
    coef = coef_bound(coef, arg.desvel_range, arg.uns4IDM_range, arg.mlp_range)
    coefp = coef_bound(coefp, arg.desvel_range, arg.uns4IDM_range, arg.mlp_range)

    # record initiation coefficients
    ini_coef = coef.clone()

    loss_log = deque(maxlen=arg.NUM_IT)
    coef_log = deque(maxlen=arg.NUM_IT)

    # Set-up Optimizer & Scheduler
    opt = torch.optim.Adam([coef], lr=arg.ADAM_LR, eps=arg.ADAM_eps, amsgrad=False)
    Sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.99)

    print('===========================================')
    print('Finding Coefficient')
    print('===========================================')

    # print(loss.grad_fn)
    for it in tqdm(range(arg.NUM_IT)):
        # Calculate loss function
        loss = Lossfn(agent, x_traj, a_traj, coef, env, arg.PI_STD, arg.NUM_SAM, w1, w2, w3, b1, b2, b3, arg)

        # Initialize the gradients calculated previous step; Clears old gradients
        opt.zero_grad()

        # Collect the data for plotting tendency
        loss_log.append(loss.data)

        # Backpropagation
        loss.backward(retain_graph=True)

        # Optimizer and Scheduler
        opt.step() # change coefficients
        Sched.step()

        # Lower and Upper Bound for Coefficient
        coef = coef_bound(coef, arg.desvel_range, arg.uns4IDM_range, arg.mlp_range)
        coef_log.append(coef.data.clone())
        coefp = coef.clone()

        # check progress
        if it%1 == 0:
            print("num_coef:{}, num:{}, loss:{}, true loss:{}, \n true_coef:{}, \n converged_coef:{}\n".format(n, it, np.round(loss.data.item(), 6), np.round(true_loss.data.item(), 6), true_coef, coef))

        if it % 1 == 0 and it > 0:
            plt.plot(loss_log)
            plt.title("it:{}".format(it))
            plt.savefig('/results/' + filename + str(n) + '_loss.png')


    # loss = Lossfn(agent, x_traj, a_traj, coef, env, arg.PI_STD, arg.NUM_SAM, w1, w2, w3, b1, b2, b3, arg)

    # end time
    toc = time.time()

    result = {'true_coef': true_coef,
              'initial_coef': ini_coef,
              'x_traj': x_traj,
              'a_traj': a_traj,
              'coef': coef,
              'coef_log': coef_log,
              'loss_log': loss_log,
              # 'loss_diff_log': loss_,
              'true_loss': true_loss,
              'filename': filename,
              'num_coef': n,
              'converging_it': it,
              'duration': toc-tic,
              'arguments': arg}

    torch.save(result, '/results/' + filename + str(n) + str(arg.NUM_coef) + "EP" + str(arg.NUM_EP) + str(np.around(arg.PI_STD, decimals=2)) + str(arg.NUM_SAM) + "IT" + str(arg.NUM_IT) + str(arg.SEED_NUMBER) +'_single_result.pkl')
    # print(result)
    return result
