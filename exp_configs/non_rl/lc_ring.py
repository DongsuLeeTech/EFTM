"""Used as an example of ring experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import IDMController, ContinuousRouter, SimLaneChangeController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoLaneChangeParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
# from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.networks.lane_change_ring import RingNetwork, ADDITIONAL_NET_PARAMS

vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {'v0': 2.5,
                                             'noise': 5.,
                                             'a': 10,
                                             'b': 10}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=0,
    num_vehicles=26)

vehicles.add(

    veh_id='rl',
    acceleration_controller=(IDMController, {'v0': 3.5}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(
    speed_mode='aggressive',
    lane_change_mode=1621,
    model='LC2013',

    ),
    initial_speed=0,
    num_vehicles=1,
)

flow_params = dict(
    # name of the experiment
    exp_tag='ring',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=3000,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    # net=NetParams(
    #     additional_params=ADDITIONAL_NET_PARAMS.copy(),
    # ),
    net=NetParams(
        additional_params={
            "length": 300,
            "lanes": 3,
            "speed_limit": 10,
            "resolution": 40,
        },
    ),
    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom3',
    ),
)
