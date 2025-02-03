import torch
from Inverse_loss import *
from Copy_Actor_Net import *

def reset_coef(desvel_range, uns4IDM_range, mlp_range):
    coef = torch.zeros(2)

    coef[0] = torch.zeros(1).uniform_(desvel_range[0])
    coef[1] = torch.zeros(1).uniform_(uns4IDM_range[0])
    coef[2] = torch.zeros(1).uniform_(mlp_range[0])

    coef = torch.cat([coef])

    return coef

def coef_bound(coef, desvel_range, uns4IDM_range, mlp_range):

    coef[0].data.clamp_(desvel_range[0],desvel_range[1])
    coef[1].data.clamp_(uns4IDM_range[0],uns4IDM_range[1])
    coef[2].data.clamp_(mlp_range[0], mlp_range[1])

    coef_copy = coef.data.clone()
    for i in [0,1]:
        if coef[i] == desvel_range[0]:
            coef[i].data.copy_(coef_copy[i] + 1e-2 * torch.rand(1).item())

        if coef[i] == uns4IDM_range[0]:
            coef[i].data.copy_(coef_copy[i] + 1e-2 * torch.rand(1).item())

    del coef_copy

    return coef

def coef_init(agent, env, arg, x_traj, a_traj, rl_des, uns4IDM, mlp):
    # true_coef = reset_coef(arg.desvel_range, arg.uns4IDM_range)

    coef = torch.zeros(3)

    coef[0] = rl_des
    coef[1] = uns4IDM
    coef[2] = mlp

    true_coef = torch.cat([coef])
    # x_traj, a_traj = trajectory_collect(args)

    w1, w2, w3, b1, b2, b3 = copy_params(agent)

    true_loss = True_Lossfn(agent, x_traj, a_traj, true_coef, env, arg.PI_STD, arg.NUM_SAM, w1, w2, w3, b1, b2, b3, arg)

    init_result = {'true_coef_log': true_coef,
                   'true_loss_log': true_loss,
                   # 'true_loss_act_log': true_loss_act,
                   'x_traj_log': x_traj,
                   'a_traj_log': a_traj,
                   }

    return init_result, true_loss