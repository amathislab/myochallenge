import numpy as np
from myosuite.envs.myo.pose_v0 import PoseEnvV0
from myosuite.envs.myo.base_v0 import BaseV0


class CustomPoseEnv(PoseEnvV0):
    def _setup(self,
            viz_site_targets:tuple = None,  # site to use for targets visualization []
            target_jnt_range:dict = None,   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value:list = None,   # desired joint vector [des_qpos]_nq
            reset_type = "init",            # none; init; random
            target_type = "generate",       # generate; switch; fixed
            obs_keys:list = PoseEnvV0.DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = PoseEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pose_thd = 0.35,
            weight_bodyname = None,
            weight_range = None,
            enable_rsi=False,
            **kwargs,
        ):
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range

        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value
        
        BaseV0._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=viz_site_targets,
                **kwargs,
                )
