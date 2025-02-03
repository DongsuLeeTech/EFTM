"""A series of reward functions."""

from gym.spaces import Box, Tuple
import numpy as np

from collections import defaultdict
from functools import reduce

def total_lc_reward(env, rl_action):
    """Put all of the reward functions we consider into list
    """
    reward_dict = {
        'rl_desired_speed': rl_desired_speed(env),
        'simple_lc_penalty': simple_lc_penalty(env),
        'dc3_penalty': follower_decel_penalty(env),
        'unsafe_penalty': unsafe_distance_penalty(env),
        'rl_action_penalty': rl_action_penalty(env, rl_action),
        'acc_penalty': punish_accelerations(env, rl_action),
        'uns4IDM_penalty': unsafe_distance_penalty4IDM(env),
        'meaningless_penalty': meaningless_penalty(env),
    }
    return reward_dict

def rl_desired_speed(env):
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids()))
    rl_des = env.initial_config.reward_params.get('rl_desired_speed', 0)

    if rl_des == 0:
        return 0

    if any(vel < -100):
        return 0.
    if len(vel) == 0:
        return 0.

    rls = env.k.vehicle.get_rl_ids()

    vel = np.array(env.k.vehicle.get_speed(rls))
    num_vehicles = len(rls)

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    eps = np.finfo(np.float32).eps

    return rl_des * (1 - (cost / (max_cost + eps)))

def unsafe_distance_penalty4IDM(env):
    uns4IDM_p = env.initial_config.reward_params.get('uns4IDM_penalty', 0)
    rls = env.k.vehicle.get_rl_ids()

    #  Parameter of IDM
    T = 1
    a = 1
    b = 1
    s0 = 2

    v = env.k.vehicle.get_speed(rls)[0]
    tw = env.k.vehicle.get_tailway(rls)[0]

    follow_id = env.k.vehicle.get_follower(rls)[0]

    if abs(tw) < 1e-3:
        tw = 1e-3

    if follow_id is None or follow_id == '':
        s_star = 0

    else:
        follow_vel = env.k.vehicle.get_speed(follow_id)
        s_star = s0 + max(
            0, follow_vel * T + follow_vel * (follow_vel - v) /
            (2 * np.sqrt(a * b)))

    rwd = uns4IDM_p * max(-5, min(0, 1 - (s_star / tw) ** 2))
    return rwd

def punish_accelerations(env,rl_action):
    acc_p = env.initial_config.reward_params.get('acc_penalty', 0)

    if rl_action is None:
        return 0
    else:
        acc = rl_action[0:1]
        mean_actions = np.mean(np.abs(np.array(acc)))
        accel_threshold = 0

        return acc_p * (accel_threshold - mean_actions)

def new_softmax(a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

def rl_action_penalty(env, actions):
    action_penalty = env.initial_config.reward_params.get('rl_action_penalty', 0)
    if actions is None or action_penalty == 0:
        return 0

    veh_id = env.k.vehicle.get_rl_ids()

    direction = actions[1::2]

    for i in range(len(direction)):
        if direction[i] <= -0.333:
            direction[i] = -1
        elif direction[i] >= 0.333:
            direction[i] = 1
        else:
            direction[i] = 0
            
    reward = 0
    if direction:
        if env.k.vehicle.get_previous_lane(veh_id) == env.k.vehicle.get_lane(veh_id):
            reward -= action_penalty
            
    return reward

def meaningless_penalty(env):
    mlp = env.initial_config.reward_params.get('meaningless_penalty', 0)
    reward = 0

    if mlp:
        for veh_id in env.k.vehicle.get_rl_ids():
            # print(env.time_counter, env.k.vehicle.get_lane(veh_id))
            if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
                lane_leaders = env.k.vehicle.get_lane_leaders(veh_id)
                headway = [(env.k.vehicle.get_x_by_id(leader) - env.k.vehicle.get_x_by_id(veh_id))
                           % env.k.network.length() / env.k.network.length() for leader in lane_leaders]
                # FOR N LANE
                if headway[env.k.vehicle.get_previous_lane(veh_id)] - headway[env.k.vehicle.get_lane(veh_id)] > 5:
                    reward -= mlp * (headway[env.k.vehicle.get_previous_lane(veh_id)])

    return reward

def simple_lc_penalty(env):
    sim_lc_penalty = env.initial_config.reward_params.get('simple_lc_penalty', 0)
    if not sim_lc_penalty:
        return 0
    reward = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
            reward -= sim_lc_penalty
    print(reward)
    return reward

def follower_decel_penalty(env):

    dc3_p = env.initial_config.reward_params.get('dc3_penalty', 0)
    if not dc3_p:
        return 0
    reward = 0
    threshold = -0.2
    rls = env.k.vehicle.get_rl_ids()
    max_decel = env.env_params.additional_params['max_decel']
    for rl in rls:
        follower = env.k.vehicle.get_follower(rl)
        if follower is not None:
            accel = env.k.vehicle.get_accel(env.k.vehicle.get_follower(rl)) or 0
            if accel < threshold:
                f = lambda x: x if x < 1 else np.log(x) + 1
                pen = dc3_p * f(abs(2 * accel / max_decel))
                reward -= pen

    return reward

def unsafe_distance_penalty(env):
    unsafe_p = env.initial_config.reward_params.get('unsafe_penalty', 0)
    rls = env.k.vehicle.get_rl_ids()
    reward = 0
    for rl in rls:
        follower = env.k.vehicle.get_follower(rl)
        if follower is not None and unsafe_p:
            tailway = env.k.vehicle.get_tailway(rl)
            gap = 5 + env.k.vehicle.get_speed(env.k.vehicle.get_follower(rl)) ** 2 / (
                        2 * env.env_params.additional_params['max_decel'])
            if tailway < gap:
                pen = unsafe_p * (gap - tailway) / gap
                reward -= pen
    return reward

