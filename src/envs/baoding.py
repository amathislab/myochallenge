import collections

import numpy as np
from myosuite.envs.myo.myochallenge.baoding_v1 import BaodingEnvV1


class CustomBaodingEnv(BaodingEnvV1):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 5.0,
        "pos_dist_2": 5.0,
        "alive": 0.0,
        "act_reg": 0.0,
    }

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        # detect fall
        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)  # keep both balls up

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2
                # Optional Keys
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                ("alive", ~is_fall),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict


import collections

import numpy as np
from myosuite.envs.myo.myochallenge.baoding_v1 import BaodingEnvV1


class CustomBaodingEnv(BaodingEnvV1):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 5.0,
        "pos_dist_2": 5.0,
        "alive": 0.0,
        "act_reg": 0.0,
    }

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        # detect fall
        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)  # keep both balls up

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2
                # Optional Keys
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                ("alive", ~is_fall),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):

        # MODIFICATION: randomnize starting target position along the cycle
        # random_phase = np.random.uniform(low=-np.pi, high=np.pi)
        random_phase = 0
        self.ball_1_starting_angle = 3.0 * np.pi / 4.0 + random_phase
        self.ball_2_starting_angle = -1.0 * np.pi / 4.0 + random_phase

        # reset counters
        self.counter = 0
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )
        # reset goal
        if time_period == None:
            time_period = self.np_random.uniform(
                low=self.goal_time_period[0], high=self.goal_time_period[1]
            )
        self.goal = (
            self.create_goal_trajectory(time_step=self.dt, time_period=time_period)
            if reset_goal is None
            else reset_goal.copy()
        )

        # reset scene (MODIFIED from base class MujocoEnv)
        qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
        qvel = self.init_qvel.copy() if reset_vel is None else reset_vel

        self.robot.reset(qpos, qvel)

        # MODIFICATION: start balls at the same position as targets
        self.step(np.zeros(39))
        qpos[23] = self.get_obs().copy()[35]
        qpos[24] = self.get_obs().copy()[36]
        qpos[30] = self.get_obs().copy()[38]
        qpos[31] = self.get_obs().copy()[39]

        self.set_state(qpos, qvel)

        return self.get_obs()
