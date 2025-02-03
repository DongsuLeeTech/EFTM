import torch
import numpy as np
from numpy import pi
from tqdm import tqdm
from Copy_Actor_Net import *

def Lossfn(agent, x_traj, a_traj, c, env, PI_STD, NUM_SAM, w1, w2, w3, b1, b2, b3, arg):
    # getLoss
    logPr = torch.zeros(1)
    logPr_act = torch.zeros(1)
    logPr_lc = torch.zeros(1)
    logPr_acc = torch.zeros(1)

    desvel, uns4IDM, mlp = torch.split(c.view(-1),1)

    env.desvel = desvel
    env.uns4IDM = uns4IDM
    env.mlp = mlp
    print('before num_it: {}, {}, {}'.format(desvel, uns4IDM, mlp))
    for num_it in range(NUM_SAM):
        logPr_ep = torch.zeros(1)
        logPr_act_ep = torch.zeros(1)
        logPr_lc_ep = torch.zeros(1)
        logPr_acc_ep = torch.zeros(1)

        t = torch.zeros(1)
        stat = x_traj[0]
        state = torch.cat([stat[:14], desvel, uns4IDM, mlp]).float()

        print('num_it: {}'.format(c))


        print('\ncalculating loss')
        for it, next_x in tqdm(enumerate(x_traj[1:3001])):
            action = Action_compute(w1, w2, w3, b1, b2, b3, arg, state)

            acc_loss = torch.zeros(1)+((action[0] - a_traj[it][0]) ** 2) / 2 / (PI_STD ** 2) + np.log(2 * pi * ((PI_STD) ** 2))

            lc = action[1:]
            proto_lc = action[1:]
            if proto_lc > 0.333:
                lc = torch.zeros(1) + 2
            elif proto_lc <= 0.333 and proto_lc>=-0.333:
                lc = torch.zeros(1) + 1
            elif proto_lc < -0.333:
                lc = torch.zeros(1)

            true_lc = a_traj[it][1] + 1

            if true_lc - lc == 0:
                lc_loss = torch.zeros(1)
            elif torch.abs(true_lc - lc) == 1:
                lc_loss = torch.zeros(1) + abs(true_lc - (proto_lc + 1))
            elif torch.abs(true_lc - lc) == 2:
                lc_loss = torch.zeros(1) + abs(true_lc - (proto_lc + 1))

            logPr_acc_ep = acc_loss.sum() + logPr_acc_ep
            logPr_lc_ep = lc_loss.sum() + logPr_lc_ep
            logPr_act_ep = acc_loss.sum() + lc_loss.sum() + logPr_act_ep
            logPr_ep = logPr_ep + acc_loss.sum() + lc_loss.sum()

            next_x = torch.cat([next_x[:14], desvel, uns4IDM, mlp]).float()
            t += 1
            state = next_x

        print('after it: {}, {}, {}, {}'.format(desvel, uns4IDM, mlp, c))
        logPr_acc += logPr_acc_ep
        logPr_lc += logPr_lc_ep
        logPr_act += logPr_act_ep
        logPr += logPr_ep
        print("acc:{}, lc:{}, logPr:{}, logPr_act:{}".format(logPr_acc, logPr_lc, logPr_ep, logPr_act))
    return logPr/NUM_SAM


def True_Lossfn(agent, x_traj, a_traj, c, env, PI_STD, NUM_SAM, w1, w2, w3, b1, b2, b3, arg):
    # getLoss
    logPr = torch.zeros(1)
    logPr_act = torch.zeros(1)
    logPr_lc = torch.zeros(1)
    logPr_acc = torch.zeros(1)

    desvel, uns4IDM, mlp = torch.split(c.view(-1),1)

    env.desvel = desvel
    env.uns4IDM = uns4IDM
    env.mlp = mlp
    print('before num_it: {}, {}, {}'.format(desvel, uns4IDM, mlp))
    for num_it in range(NUM_SAM):
        logPr_ep = torch.zeros(1)
        logPr_act_ep = torch.zeros(1)
        logPr_lc_ep = torch.zeros(1)
        logPr_acc_ep = torch.zeros(1)

        t = torch.zeros(1)

        stat = x_traj[0]
        state = torch.cat([stat[:14], desvel, uns4IDM, mlp]).float()
        print('num_it: {}, {}, {}'.format(desvel, uns4IDM, mlp))

        print('\ncalculating loss')
        for it, next_x in tqdm(enumerate(x_traj[1:3001])):
            action = Action_compute(w1, w2, w3, b1, b2, b3, arg, state)

            acc_loss = torch.zeros(1) + ((action[0] - a_traj[it][0]) ** 2) / 2 /(PI_STD ** 2) \
                       + np.log(2 * pi * ((PI_STD) ** 2))

            proto_lc = action[1:]
            if proto_lc > 0.333:
                lc = torch.zeros(1) + 2
            elif proto_lc <= 0.333 and proto_lc >= -0.333:
                lc = torch.zeros(1) + 1
            elif proto_lc < -0.333:
                lc = torch.zeros(1)

            true_lc = a_traj[it][1] + 1

            if true_lc - lc == 0:
                lc_loss = torch.zeros(1)
            elif torch.abs(true_lc - lc) == 1:
                lc_loss = torch.zeros(1) + true_lc - proto_lc
            elif torch.abs(true_lc - lc) == 2:
                lc_loss = torch.zeros(1) + true_lc - proto_lc

            logPr_acc_ep = acc_loss.sum() + logPr_acc_ep
            logPr_lc_ep = lc_loss.sum() + logPr_lc_ep
            logPr_act_ep = acc_loss.sum() + lc_loss.sum() + logPr_act_ep
            logPr_ep = logPr_ep + acc_loss.sum() + lc_loss.sum()


            next_x = torch.cat([next_x[:14], desvel, uns4IDM, mlp]).float()
            t += 1
            state = next_x

        print('after it: {}, {}, {}, {}'.format(desvel, uns4IDM, mlp, c))
        logPr_acc += logPr_acc_ep
        logPr_lc += logPr_lc_ep
        logPr_act += logPr_act_ep
        logPr += logPr_ep
        print("acc:{}, lc:{}, logPr:{}, logPr_act:{}".format(logPr_acc, logPr_lc, logPr_ep, logPr_act))
    return logPr/NUM_SAM