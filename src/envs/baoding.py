import collections
import random

import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myochallenge.baoding_v1 import (WHICH_TASK,
                                                       BaodingEnvV1, Task)


class CustomBaodingEnv(BaodingEnvV1):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 5.0,
        "pos_dist_2": 5.0,
        "alive": 0.0,
        "act_reg": 0.0,
        # "palm_up": 0.0,
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

        # # rewards for keeping palm up: negative for deviations
        # pronation_reward = np.exp(-(self.get_obs()[0] + 1.57) * 5) - 0.5
        # flexion_reward = np.exp(-abs(self.get_obs()[2]) * 2) - 0.5
        # palm_up_reward = pronation_reward + flexion_reward

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
                # ("palm_up", palm_up_reward),
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

    def _init_targets_with_balls(self) -> None:
        desired_angle_wrt_palm = self.goal[self.counter].copy()
        desired_angle_wrt_palm[0] = desired_angle_wrt_palm[0] + self.ball_1_starting_angle
        desired_angle_wrt_palm[1] = desired_angle_wrt_palm[1] + self.ball_2_starting_angle

        desired_positions_wrt_palm = [0,0,0,0]
        desired_positions_wrt_palm[0] = self.x_radius*np.cos(desired_angle_wrt_palm[0]) + self.center_pos[0]
        desired_positions_wrt_palm[1] = self.y_radius*np.sin(desired_angle_wrt_palm[0]) + self.center_pos[1]
        desired_positions_wrt_palm[2] = self.x_radius*np.cos(desired_angle_wrt_palm[1]) + self.center_pos[0]
        desired_positions_wrt_palm[3] = self.y_radius*np.sin(desired_angle_wrt_palm[1]) + self.center_pos[1]

        # update both sims with desired targets
        for sim in [self.sim, self.sim_obsd]:
            sim.model.site_pos[self.target1_sid, 0] = desired_positions_wrt_palm[0]
            sim.model.site_pos[self.target1_sid, 1] = desired_positions_wrt_palm[1]
            sim.model.site_pos[self.target2_sid, 0] = desired_positions_wrt_palm[2]
            sim.model.site_pos[self.target2_sid, 1] = desired_positions_wrt_palm[3]
            sim.forward()

    def _add_noise_to_palm_position(self, qpos: np.ndarray, noise: float = 1) -> np.ndarray:
        assert 0 <= noise <= 1, "Noise must be between 0 and 1"

        # pronation-supination of the wrist
        # noise = 1 corresponds to 10 degrees from facing up (one direction only)
        qpos[0] = self.np_random.uniform(low= -np.pi/2, high = -np.pi/2 + np.pi/18 * noise)

        # ulnar deviation of wrist: 
        # noise = 1 corresponds to 10 degrees on either side
        qpos[1] = self.np_random.uniform(low= -np.pi/18 * noise, high = np.pi/18 * noise)

        # extension flexion of the wrist
        # noise = 1 corresponds to 10 degrees on either side
        qpos[2] = self.np_random.uniform(low= -np.pi/18 * noise, high = np.pi/18 * noise)

        return qpos

    def _add_noise_to_finger_positions(self, qpos: np.ndarray, noise: float = 1) -> np.ndarray:
        assert 0 <= noise <= 1, "Noise parameter must be between 0 and 1"
        
        # thumb all joints
        # noise = 1 corresponds to 10 degrees on either side
        qpos[3:7] = self.np_random.uniform(low= -np.pi/18 * noise, high = np.pi/18 * noise)
        
        # finger joints
        # noise = 1 corresponds to 30 degrees bent instead of fully open
        qpos[[7,9,10,11,13,14,15,17,18,19,21,22]] = self.np_random.uniform(low=0, high=np.pi/6 * noise)

        # finger abduction (sideways angle)
        # noise = 1 corresponds to 5 degrees on either side
        qpos[[8,12,16,20]] = self.np_random.uniform(low= -np.pi/36 * noise, high= np.pi/36 * noise)

        return qpos

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):
        self.which_task = self.sample_task()
        if self.rsi:
            # MODIFICATION: randomize starting target position along the cycle
            random_phase = np.random.uniform(low=-np.pi, high=np.pi)
        else:
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
        print(qpos)
        self.robot.reset(qpos, qvel)

        if self.rsi:
            if np.random.uniform(0,1)<self.rsi_probability:
                # self._init_targets_with_balls()
                self.step(np.zeros(39))

                # update ball positions
                obs = self.get_obs().copy()
                qpos[23] = obs[35]  # ball 1 x-position
                qpos[24] = obs[36]  # ball 1 y-position
                qpos[30] = obs[38]  # ball 2 x-position
                qpos[31] = obs[39]  # ball 2 y-position

        if self.noise_balls:
            # update balls x,y,z positions with relative noise
            for i in [23,24,25,30,31,32]:
                qpos[i] += np.random.uniform(
                    low = -self.noise_balls,
                    high = self.noise_balls) 

        if self.noise_palm:
            qpos = self._add_noise_to_palm_position(qpos, self.noise_palm)

        if self.noise_fingers:
            qpos = self._add_noise_to_finger_positions(qpos, self.noise_fingers)
        
        if self.rsi or self.noise_palm or self.noise_fingers or self.noise_balls:
            self.set_state(qpos, qvel)

        return self.get_obs()

    def _setup(
        self,
        frame_skip: int = 10,
        drop_th=1.25,  # drop height threshold
        proximity_th=0.015,  # object-target proximity threshold
        goal_time_period=(5, 5),  # target rotation time period
        goal_xrange=(0.025, 0.025),  # target rotation: x radius (0.03)
        goal_yrange=(0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
        obs_keys: list = BaodingEnvV1.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        task=None,
        enable_rsi=False,  # random state init for balls
        noise_palm=0,      # magnitude of noise for palm (between 0 and 1)
        noise_fingers=0,   # magnitude of noise for fingers (between 0 and 1)
        noise_balls=0,   # relative magnitude of noise for the balls (1 is 100% relative noise)
        rsi_probability=1, #probability of implementing RSI
        **kwargs
    ):

        # user parameters
        self.task = task
        self.which_task = self.sample_task()
        self.rsi = enable_rsi
        self.noise_palm = noise_palm
        self.noise_fingers = noise_fingers
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange
        self.noise_balls = noise_balls
        self.rsi_probability = rsi_probability

        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left
        self.ball_1_starting_angle = 3.0 * np.pi / 4.0
        self.ball_2_starting_angle = -1.0 * np.pi / 4.0

        # init desired trajectory, for rotations
        self.center_pos = [-0.0125, -0.07]  # [-.0020, -.0522]
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        self.counter = 0
        self.goal = self.create_goal_trajectory(
            time_step=frame_skip * self.sim.model.opt.timestep, time_period=6
        )

        # init target and body sites
        self.object1_sid = self.sim.model.site_name2id("ball1_site")
        self.object2_sid = self.sim.model.site_name2id("ball2_site")
        self.object1_gid = self.sim.model.geom_name2id("ball1")
        self.object2_gid = self.sim.model.geom_name2id("ball2")
        self.target1_sid = self.sim.model.site_name2id("target1_site")
        self.target2_sid = self.sim.model.site_name2id("target2_site")
        self.sim.model.site_group[self.target1_sid] = 2
        self.sim.model.site_group[self.target2_sid] = 2

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )

        # reset position
        self.init_qpos[:-14] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.57  # Palm up
        
    def sample_task(self):
        if self.task is None:
            return Task(WHICH_TASK)
        else:
            if self.task == "cw":
                return Task(Task.BAODING_CW)
            elif self.task == "ccw":
                return Task(Task.BAODING_CCW)
            elif self.task == "random":
                return Task(random.choice(list(Task)))
            else:
                raise ValueError("Unknown task for baoding: ", self.task)
