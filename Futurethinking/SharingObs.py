import numpy as np

def sharing_obs(networks, vehicles, agent, ids):

    agent = agent

    observation = []
    max_speed = networks.max_speed()
    length = networks.length()
    max_lanes = max(
        networks.num_lanes(edge)
        for edge in networks.get_edge_list())

    for j in agent:
        rl_speed = vehicles.get_speed(j)
        lane_leaders = vehicles.get_lane_leaders(j)
        lane_followers = vehicles.get_lane_followers(j)

        for i in range(0, max_lanes):

            if vehicles.get_lane(j) == i:
                lane_followers_speed = vehicles.get_lane_followers_speed(j)[max(0, i - 1):i + 2]
                lane_leaders_speed = vehicles.get_lane_leaders_speed(j)[max(0, i - 1):i + 2]

                lane_followers_pos = [vehicles.get_x_by_id(follower) for follower in lane_followers]
                lane_leaders_pos = [vehicles.get_x_by_id(leader) for leader in lane_leaders]

                lane_leaders_pos = lane_leaders_pos[max(0, i - 1):i + 2]
                lane_followers_pos = lane_followers_pos[max(0, i - 1):i + 2]

                if i == 0:
                    f_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_followers_speed]
                    f_sp.insert(0, -1.)
                    l_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_leaders_speed]
                    l_sp.insert(0, -1.)
                    f_pos = [((vehicles.get_x_by_id(j) - pos) % length / length)
                             for pos in lane_followers_pos]
                    f_pos.insert(0, -1.)
                    l_pos = [(pos - vehicles.get_x_by_id(j)) % length / length
                             for pos in lane_leaders_pos]
                    l_pos.insert(0, -1.)
                    lanes = [0.]

                elif i == max_lanes - 1:
                    f_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_followers_speed]
                    f_sp.insert(2, -1.)
                    l_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_leaders_speed]
                    l_sp.insert(2, -1.)
                    f_pos = [((vehicles.get_x_by_id(j) - pos) % length / length)
                             for pos in lane_followers_pos]
                    f_pos.insert(2, -1.)
                    l_pos = [(pos - vehicles.get_x_by_id(j)) % length / length
                             for pos in
                             lane_leaders_pos]
                    l_pos.insert(2, -1.)
                    lanes = [1.]

                else:
                    f_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_followers_speed]
                    l_sp = [(speed - rl_speed) / max_speed
                            for speed in lane_leaders_speed]
                    f_pos = [((vehicles.get_x_by_id(j) - pos) % length / length)
                             for pos in lane_followers_pos]
                    l_pos = [(pos - vehicles.get_x_by_id(j)) % length / length
                             for pos in lane_leaders_pos]
                    lanes = [0.5]

                rl_sp = [rl_speed / max_speed]
                positions = l_pos + f_pos
                speeds = rl_sp + l_sp + f_sp
                obs = np.array(speeds + positions + lanes)

        observation.append(obs)

    return observation