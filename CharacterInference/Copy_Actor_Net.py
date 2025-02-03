import torch
import torch.nn as nn
import torch.nn.functional as F

def copy_params(agent):
    weights = agent.get_weights()
    for key, val in weights.items():
        weights_val = val

    wei_vlist = []
    for i in weights_val.items():
        wei_vlist.append(i)

    policy_wei = []
    policy_bias = []
    for i in range(6):
        if i % 2 == 0:
            policy_wei.append(wei_vlist[i])
        else:
            policy_bias.append(wei_vlist[i])


    w1 = torch.from_numpy(policy_wei[0][1]).float()
    w2 = torch.from_numpy(policy_wei[1][1]).float()
    w3 = torch.from_numpy(policy_wei[2][1]).float()
    b1 = torch.from_numpy(policy_bias[0][1]).float()
    b2 = torch.from_numpy(policy_bias[1][1]).float()
    b3 = torch.from_numpy(policy_bias[2][1]).float()

    return w1, w2, w3, b1, b2, b3

def Action_compute(w1, w2, w3, b1, b2, b3, arg, state):

    torch.manual_seed(arg.SEED_NUMBER)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(arg.SEED_NUMBER)
    # input to hidden 1
    out = F.relu(F.linear(state.float(), w1, b1))
    # hidden 1 to hidden 2
    out = F.relu(F.linear(out, w2, b2))
    # hidden 2 to output
    out = torch.tanh(F.linear(out, w3, b3))

    return out

