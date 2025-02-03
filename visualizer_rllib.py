"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::
        python ./Q_visualizer.py /tmp/ray/result_dir 1
parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import gym
import numpy as np
import os
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import ray

from Futurethinking.ActorNet import *
from Futurethinking.SharingObs import *
from Futurethinking.FindNearest import *

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

EXAMPLE_USAGE = """
example usage:
    python ./Q_visualizer.py /ray_results/experiment_dir/result_dir 1
    
Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def visualizer_rllib(args, c_list):
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    name = result_dir.split("/")[-2:]

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
    c_list = c_list

    flow_params['initial'].reward_params['rl_desired_speed'] = [c_list[0] if k % 3 == 0 else c_list[3]
                                                                if k % 3 == 1 else c_list[6]
                                                                for k in range(21)]
    flow_params['initial'].reward_params['uns4IDM_penalty'] = [c_list[1] if k % 3 == 0 else c_list[4]
                                                               if k % 3 == 1 else c_list[7]
                                                               for k in range(21)]
    flow_params['initial'].reward_params['meaningless_penalty'] = [c_list[2] if k % 3 == 0 else c_list[5]
                                                                   if k % 3 == 1 else c_list[8]
                                                                   for k in range(21)]
    flow_params['initial'].reward_params['target_velocity'] = [3.7 if k % 3 == 0 else 3.6 if k % 3 == 1 else 3.8
                                                               for k in range(21)]

    eftm = args.eftm
    print('===================================================')
    print(eftm)
    print('===================================================')

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
            print('Q_visualizer.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('Q_visualizer.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./Q_visualizer.py /tmp/ray/result_dir 1 --run PPO')
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
        sim_params.save_render = False

    # Create and register a gym+rllib env
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
    w1, w2, w3, b1, b2, b3 = copy_params(agent)

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
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn']
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

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    rl_speed = []  # store rl controlled vehicle's speed

    # log2_stack = defaultdict(list)  # This dict stores log2 data during rollouts
    totalreturn = 0

    if args.evaluate:
        env.unwrapped.env_params.evaluate = True    # To cover bug

    for i in range(args.num_rollouts):
        vel = []
        vel_dict = defaultdict(list)
        timerange = []
        timestep = 0.1
        state = env.reset()
        ret = 0
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0
        for _ in range(env_params.horizon):
            #  env parameter
            vehicles = env.unwrapped.k.vehicle
            networks = env.unwrapped.k.network

            #  vehicles' ID
            ids = vehicles.get_ids()
            main_ids = vehicles.get_rl_ids()[0]
            rls = vehicles.get_rl_ids()

            speeds = vehicles.get_speed(ids)
            timerange.append(vehicles.get_timestep(ids[-1]) / 10000)

            max_speed = networks.max_speed()
            length = networks.length()
            num_veh = len(vehicles.get_ids())
            lane_number = max(
                networks.num_lanes(edge) for edge in networks.get_edge_list())

            #  Vehicles' ID that the main agent can observe
            #  (i.e., Vehicles where in sensing space)
            sensible_veh = [i for i in ids if abs(vehicles.get_x_by_id(main_ids) - vehicles.get_x_by_id(i)) <= length]

            #  Sharing observation of vehicles in which sensible space
            shared_obs = sharing_obs(networks, vehicles, sensible_veh, main_ids)

            #  Vehicle characteristic
            veh_char = [np.array([env.initial_config.reward_params['rl_desired_speed'][i], env.initial_config.reward_params['uns4IDM_penalty'][i],
                         env.initial_config.reward_params['meaningless_penalty'][i]]) for i in range(len(ids))
                        if ids[i] in sensible_veh]

            target_velo = flow_params['initial'].reward_params['target_velocity']
            if eftm == 'NOR':
                pass
            elif eftm == 'IRC':
                # combination of observation and characteristic
                obs_w_char = [np.concatenate([shared_obs[i], veh_char[i]/3, [target_velo[i]/max_speed]]) for i in
                              range(len(target_velo))]
            elif eftm == 'FCE':
                # FCE
                obs_w_char = [np.concatenate([shared_obs[i], veh_char[0]/3, [target_velo[i] / max_speed]]) for i in
                              range(len(target_velo))]

            if eftm != 'NOR':
                # Action prediction
                from Futurethinking.Configurations import futurethinking_config as arg
                pred_A = [Action_compute(w1, w2, w3, b1, b2, b3, arg, i) for i in obs_w_char]
                # Dictionary Action with vehicle ID
                dict_A_ID = {x: y for x, y in zip(sensible_veh, pred_A)}

                # Sensing Space Prediction
                pred_next_OS = []
                for i in range(num_veh):
                    tmp_ID = vehicles.get_ids()[i]
                    if tmp_ID in sensible_veh:
                        next_vel = np.clip(vehicles.get_speed(tmp_ID) + timestep * dict_A_ID[tmp_ID][0], 0, max_speed)
                        next_pos = np.clip(vehicles.get_x_by_id(tmp_ID) + timestep * vehicles.get_speed(tmp_ID), 0, length)
                        next_lan = np.clip(vehicles.get_lane(tmp_ID) + dict_A_ID[tmp_ID][1], 0, lane_number - 1)
                        next_OS = [next_vel, next_pos, next_lan]

                        pred_next_OS.append(next_OS)

                    else:
                        pred_next_OS.append([-100, -100, -100])

                # Observation prediction
                S_veh = []
                L_veh = []
                R_veh = []
                for i in range(len(pred_next_OS)):
                    if vehicles.get_lane(main_ids)-1 == pred_next_OS[i][2]:
                        R_veh.append(pred_next_OS[i])
                    elif vehicles.get_lane(main_ids) == pred_next_OS[i][2]:
                        S_veh.append(pred_next_OS[i])
                    elif vehicles.get_lane(main_ids)+1 == pred_next_OS[i][2]:
                        L_veh.append(pred_next_OS[i])
                    else:
                        pass

                # Select the observable vehicles at time t+1
                # ma_state = [vehicles.get_speed(main_ids), vehicles.get_x_by_id(main_ids), vehicles.get_lane(main_ids)]
                pred_next_obs_R = []
                pred_next_obs_S = []
                pred_next_obs_L = []
                pred_next_obs_R.append(find_nearest(R_veh, vehicles.get_x_by_id(main_ids), length))
                pred_next_obs_S.append(find_nearest(S_veh, vehicles.get_x_by_id(main_ids), length))
                pred_next_obs_L.append(find_nearest(L_veh, vehicles.get_x_by_id(main_ids), length))

                pred_vel = [pred_next_obs_R[0][0][0], pred_next_obs_S[0][0][0], pred_next_obs_L[0][0][0],
                            pred_next_obs_R[0][1][0], pred_next_obs_S[0][1][0], pred_next_obs_L[0][1][0]]
                pred_pos_l = [pred_next_obs_R[0][0][1], pred_next_obs_S[0][0][1], pred_next_obs_L[0][0][1]]
                pred_pos_f = [pred_next_obs_R[0][1][1], pred_next_obs_S[0][1][1], pred_next_obs_L[0][1][1]]

                # Calculate the observation at time t+1
                pred_next_obs = np.concatenate([np.array(
                    [vehicles.get_speed(main_ids) / max_speed] + # main agent speed
                    [-1. if speed < -50 else (speed - vehicles.get_speed(main_ids)) / max_speed
                       for speed in pred_vel] +  # velocity
                    [-1. if pos < -50 else (pos - vehicles.get_x_by_id(main_ids)) % length / length
                       for pos in pred_pos_l] +
                    [-1. if pos < -50 else (vehicles.get_x_by_id(main_ids) - pos) % length / length
                                              for pos in pred_pos_f] +
                    [0.5 if vehicles.get_lane(main_ids) != 0 or (lane_number-1)
                     else vehicles.get_lane(main_ids)/(lane_number-1)]), # RL lane_number
                    veh_char[0],
                    [target_velo[0] / max_speed]])

            # only include non-empty speeds
            if speeds:
                vel.append(np.mean(speeds))
                for veh_id, speed in zip(ids, speeds):
                    vel_dict[veh_id].append(speed)

            if multiagent:
                state2 = state
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                                state[agent_id], state=state_init[agent_id],
                                policy_id=policy_map_fn(agent_id))
                    else:
                        if eftm != 'NOR':
                            state2[main_ids] = pred_next_obs

                        # Actor action
                        out = F.relu(F.linear(torch.from_numpy(state2[agent_id]).float(), w1, b1))
                        out = F.relu(F.linear(out, w2, b2))
                        act = torch.tanh(F.linear(out, w3, b3))

                        action[agent_id] = act.numpy()

                        if action[agent_id][1] > 1 / 3:
                            action[agent_id][1] = 1
                        elif action[agent_id][1] < -1 / 3:
                            action[agent_id][1] = -1
                        elif 1 / 3 > action[agent_id][1] > -1 / 3:
                            action[agent_id][1] = 0

            else:
                action = agent.compute_action(state)

            state, reward, done, _ = env.step(action)

            totalreturn = reward[main_ids] + totalreturn

            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

        print('Total Return: {}'.format(totalreturn))
        ind_rew_list.append(totalreturn)

        if multiagent:
            for key in rets.keys():
                rets[key].append(ret[key])
        else:
            rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        if multiagent:
            for agent_id, rew in rets.items():
                print('Round {}, Return: {} for agent {}'.format(
                    i, ret.values(), agent_id))
                col_rew_list.append(ret['av'][0])
        else:
            print('Round {}, Return: {}'.format(i, ret))

        rl_speed = [np.mean(vel_dict[rl]) for rl in vehicles.get_rl_ids()]

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
    parser.add_argument(
        '--eftm',
        type=str,
        default='NOR',
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)

    seed = 1500
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    ind_rew_list = []
    col_rew_list = []
    redundancy = []
    for l in range(125):
        c11 = random.choice([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        c21 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c31 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c12 = random.choice([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        c31 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c12 = random.choice([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        c22 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c32 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c13 = random.choice([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        c23 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c33 = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.0])
        c_list = [c11, c21, c31, c12, c22, c32, c13, c23, c33]

        redundancy.append(c_list)
        visualizer_rllib(args, c_list)
