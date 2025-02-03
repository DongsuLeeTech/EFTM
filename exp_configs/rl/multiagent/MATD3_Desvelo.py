"""Ring road example.

Trains a number of autonomous vehicles to stabilize the flow of 22 vehicles in
a variable length ring road.
"""
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter,  SimLaneChangeController
from flow.envs.multiagent.ring.MADLC_Des_Ring import *
# from flow.networks.ring import RingNetwork
from flow.networks.lane_change_ring import RingNetwork
from flow.utils.registry import make_create_env

import numpy as np
import random
import math
import os

# name of this file
current_file_name_py = os.path.abspath(__file__).split('/')[-1]
# remove file extension
current_file_name = current_file_name_py[:-3]

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 30
# number of parallel workers
N_CPUS = 30
# number of automated vehicles. Must be less than or equal to 22.
NUM_AUTOMATED = 1


# We evenly distribute the automated vehicles in the network.
num_human = 19 - NUM_AUTOMATED
humans_remaining = num_human

vehicles = VehicleParams()
for i in range(NUM_AUTOMATED):
    # Add a fraction of the remaining human vehicles.
    vehicles_to_add = round(humans_remaining / (NUM_AUTOMATED - i))
    humans_remaining -= 2 * vehicles_to_add
    for j in range(10):
        vehicles.add(
            veh_id="human_{}".format(j),
            acceleration_controller=(IDMController, {
                'v0': random.sample([k for k in np.arange(3.0, 4.0, 0.2)], 1)[-1],
                "noise": 0.2
            }),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            routing_controller=(ContinuousRouter, {}),
            initial_speed=0,
            num_vehicles=1)

        vehicles.add(
            veh_id="lc_{}".format(j),
            acceleration_controller=(IDMController, {
                'v0': random.sample([k for k in np.arange(3.5, 4.0, 0.2)], 1)[-1],
                "noise": 0.2
            }),
            car_following_params=SumoCarFollowingParams(
                min_gap=0
            ),
            routing_controller=(ContinuousRouter, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            lane_change_params=SumoLaneChangeParams(
                speed_mode='aggressive',
                lane_change_mode=1621,
                model='LC2013',
            ),
            initial_speed=0,
            num_vehicles=1)

        # Add one automated vehicle.
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        initial_speed=0,
        num_vehicles=1)


flow_params = dict(
    # name of the experiment
    exp_tag=current_file_name,

    # random seed
    seed=1004,

    # name of the flow environment the experiment is running on
    env_name=TD3MADLC_DESAccelPOEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "target_velocity": 4.,
            "ring_length": [260, 310],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 300,
            "lanes": 5, # Too
            "speed_limit": 5,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='random',
        # perturbation=3,  # Too big to fail
        # shuffle=True,
        # bunching=100,
        reward_params={
            'rl_desired_speed': 0.,
            'simple_lc_penalty': 0,
            'rl_action_penalty': .3,
            'unsafe_penalty': 0,
            'dc3_penalty': 0,
            'uns4IDM_penalty': 0.,
            'acc_penalty': 0,
            'meaningless_penalty': 0.,
            'target_velocity': 0.,
        },
    ),
)


create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return DDPGTorchPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
