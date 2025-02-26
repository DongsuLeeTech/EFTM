"""Contains an experiment class for running simulations."""
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
import datetime
import logging
import time
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict

class Experiment:

    def __init__(self, flow_params, custom_callables=None):

        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        """Run the given network for a set number of runs.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict < str, Any >
            contains returns, average speed per step
        """
        num_steps = self.env.env_params.horizon

        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        # used to store
        info_dict = {
            "returns": [],
            "velocities": [],
            "outflows": [],
        }
        info_dict.update({
            key: [] for key in self.custom_callables.keys()
        })

        if rl_actions is None:
            def rl_actions(*_):
                return None

        # time profiling information
        t = time.time()
        times = []

        ''' Experiment Operation Part'''
        for i in range(num_runs):
            ret = 0
            timerange = []
            vel = []
            var_list = []
            vel_dict = defaultdict(list)
            custom_vals = {key: [] for key in self.custom_callables.keys()}
            state = self.env.reset()
            for j in range(num_steps):
                vehicles = self.env.unwrapped.k.vehicle
                ids = vehicles.get_ids()

                t0 = time.time()
                state, reward, done, _ = self.env.step(rl_actions(state))
                t1 = time.time()
                times.append(1 / (t1 - t0))
                timerange.append(vehicles.get_timestep(ids[-1]))

                speeds = self.env.k.vehicle.get_speed(ids)

                if speeds:
                    vel.append(np.mean(speeds))
                    var_list.append(np.var(speeds))

                    for veh_id, speed in zip(ids, speeds):
                        vel_dict[veh_id].append(speed)
                f = open("/home/bmil/ray_results/DLCF_TD3/IEEE/27LC2013_50.csv", 'a')
                f.write('{}\n'.format(speeds))
                f.close()
                accs = [self.env.k.vehicle.get_realized_accel(id) for id in ids]

                # if accs:
                #     for veh_id, ACC in zip(ids, accs):
                #         acc_dict[veh_id].append(ACC)
                # f = open("/home/bmil/ray_results/DLCF_TD3/AAAI/LC2013_ACC05.csv", 'a')
                # f.write('{}\n'.format(accs))
                # f.close()

                # Compute the velocity speeds and cumulative returns.
                # vel.append(np.mean(self.env.k.vehicle.get_speed(ids)))
                ret += reward

                # Compute the results for the custom callables.
                for (key, lambda_func) in self.custom_callables.items():
                    custom_vals[key].append(lambda_func(self.env))
                if done:
                    break
                # if j % 1000 == 20:
                #     print(self.env.get_state())

            # End Operation##################################################
            # Store the information from the run in info_dict.
            outflow = self.env.k.vehicle.get_outflow_rate(int(500))
            info_dict["returns"].append(ret)
            info_dict["velocities"].append(np.mean(vel))
            info_dict["outflows"].append(outflow)
            for key in custom_vals.keys():
                info_dict[key].append(np.mean(custom_vals[key]))
            print("Round {0}, return: {1}".format(i, ret))

        # BMIL EDIT FOR PLOT DATA
        veh = list(vel_dict.keys())
        plt.subplot(3, 1, 1)
        plt.title('SimulationResults')
        for v in veh[:-1]:
            plt.plot(timerange, vel_dict[v])
        plt.xlabel('timestep(s)')
        plt.ylabel('velocity(m/s)')
        # plt.axis([110000, 375000, 4.3, 5.0])
        # plt.legend(veh[-14:-8], fontsize=9, loc='upper right')
        plt.grid(True)
        plt.show()

        # Print the averages/std for all variables in the info_dict.
        for key in info_dict.keys():
            print("Average, std {}: {}, {}".format(
                key, np.mean(info_dict[key]), np.std(info_dict[key])))

        print("Total time:", time.time() - t)
        print("steps/second:", np.mean(times))
        self.env.terminate()

        if convert_to_csv and self.env.simulator == "traci":
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            dir_path = self.env.sim_params.emission_path
            emission_filename = \
                "{0}-emission.xml".format(self.env.network.name)
            emission_path = os.path.join(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)

        return info_dict
