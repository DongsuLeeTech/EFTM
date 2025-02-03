import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import numpy as np
import os
import sys
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

import ray

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

from flow.envs import LCIAccelPOEnv

from tqdm import tqdm

from Inverse_exp import exp_inverse
from Inverse_config import Inverse_config
from Inverse_coef import coef_init, coef_bound, reset_coef
from Inverse_loss import Lossfn
from Copy_Actor_Net import *


def Trajectory_collect(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    name = result_dir.split("/")[-2:]
    filename = str(result_dir.split("/")[-2:])

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    #  setting coef
    flow_params['initial'].reward_params['rl_desired_speed'] = 1.9
    flow_params['initial'].reward_params['uns4IDM_penalty'] = 0.3
    flow_params['initial'].reward_params['meaningless_penalty'] = 0.9

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # for hacks for old pkl files TODO: remove eventually
    if not hasattr(sim_params, 'use_ballistic'):
        sim_params.use_ballistic = False

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('Trajectory_collect.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('Trajectory_collect.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./Trajectory_collect.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if args.gen_emission else None

    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = False  # will be set to True below
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        if args.render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True

    # Create and register a gym + rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True  # FIXME: this not works

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon


    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    # Parameters extraction for model copy
    weights = agent.get_weights()
    for key, val in weights.items():
        weights_val = val

    wei_vlist = []
    for i in weights_val.items():
        wei_vlist.append(i)

    policy_wei = []
    policy_bias = []
    for i in range(6):
        if i % 2 == 0 :
            policy_wei.append(wei_vlist[i])
        else:
            policy_bias.append(wei_vlist[i])

    # Inverse Argument
    arg = Inverse_config()

    # Random Seed
    random.seed(arg.SEED_NUMBER)
    torch.manual_seed(arg.SEED_NUMBER)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(arg.SEED_NUMBER)
    np.random.seed(arg.SEED_NUMBER)

    # set-up cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CPU or GPU
    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)

    if args.render_mode == 'sumo_gui':
        env.sim_params.render = False  # set to True after initializing agent and env

    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn'].func
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn'].func
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    def new_softmax(a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    if args.evaluate:
        env.unwrapped.env_params.evaluate = True  # To cover bug
    episode = 0
    while episode <= args.num_rollouts:
        episode += 1
        x_traj = []
        a_traj = []
        state = env.reset()
        ts = torch.zeros(1)

        # print(state)
        while ts < env_params.horizon:

            # Collect State
            x_traj.append(state)
            torch.set_printoptions(precision=4)
            # Greedy action
            out = F.relu(F.linear(torch.from_numpy(state).float(), torch.from_numpy(policy_wei[0][1]),
                                  torch.from_numpy(policy_bias[0][1])))
            out = F.relu(F.linear(out, torch.from_numpy(policy_wei[1][1]),
                                  torch.from_numpy(policy_bias[1][1])))
            out = torch.tanh(F.linear(out, torch.from_numpy(policy_wei[2][1]),
                                      torch.from_numpy(policy_bias[2][1])))

            action = out.numpy()

            # Compute Lane Change - For Collect Action
            lc = action[1:]

            if lc >= 0.333:
                lc = np.array([1.])
            elif lc <= -0.333:
                lc = np.array([-1.])
            else:
                lc = np.array([0.])

            actions = np.concatenate([action[:1], lc])
            a_traj.append(np.round(actions, 4))

            w1, w2, w3, b1, b2, b3 = copy_params(agent)
            # torch.set_printoptions(sci_mode=True)
            # print(torch.from_numpy(state).float())
            Action = Action_compute(w1, w2, w3, b1, b2, b3, arg, torch.Tensor(state))

            # Compute next state
            state, reward, done, _ = env.step(actions)

            ts +=1

    x_traj = torch.Tensor(x_traj).float()
    a_traj = torch.Tensor(a_traj).float()

    rl_des = env.initial_config.reward_params.get('rl_desired_speed', 0) / 3
    uns4IDM = env.initial_config.reward_params.get('uns4IDM_penalty', 0) / 3
    mlp = env.initial_config.reward_params.get('meaningless_penalty', 0) / 3
    true_coef = torch.Tensor([rl_des, uns4IDM, mlp])
    print(rl_des, uns4IDM, mlp, true_coef)

    f_name = 'INVERSE{},{}'.format(rl_des, uns4IDM)

    init_result, true_loss = coef_init(agent, env, arg, x_traj, a_traj, rl_des, uns4IDM, mlp)
    n = 3
    result = exp_inverse(true_coef, arg, env, agent, x_traj, a_traj, true_loss, f_name, 3)

    torch.save(result, '/data/190309_2000IT' + filename + str(n) + str(arg.NUM_coef) + "EP" + str(
        arg.NUM_EP) + str(np.around(arg.PI_STD, decimals=2)) + str(arg.NUM_SAM) + "IT" + str(arg.NUM_IT) + str(
        arg.SEED_NUMBER) + '_result.pkl')

    print('------------------------------------')
    print('Inverse end')
    print('------------------------------------')

    # terminate the environment
    env.unwrapped.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    Trajectory_collect(args)
