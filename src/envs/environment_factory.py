import gym


class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        # make myosuite envs
        if env_name == "MyoFingerPoseFixed":
            return gym.make("myoFingerPoseFixed-v0")
        elif env_name == "MyoFingerPoseRandom":
            return gym.make("myoFingerPoseRandom-v0")
        elif env_name == "MyoFingerReachFixed":
            return gym.make("myoFingerReachFixed-v0")
        elif env_name == "MyoFingerReachRandom":
            return gym.make("myoFingerReachRandom-v0")
        elif env_name == "MyoHandKeyTurnFixed":
            return gym.make("myoHandKeyTurnFixed-v0")
        elif env_name == "MyoHandKeyTurnRandom":
            return gym.make("myoHandKeyTurnRandom-v0")
        elif env_name == "MyoBaodingBallsP1":
            return gym.make("myoChallengeBaodingP1-v1")
        elif env_name == "CustomMyoBaodingBallsP1":
            return gym.make("CustomMyoChallengeBaodingP1-v1", **kwargs)
        elif env_name == "CustomMyoReorientP1":
            return gym.make("CustomMyoChallengeDieReorientP1-v0", **kwargs)
        elif env_name == "CustomMyoReorientP2":
            return gym.make("CustomMyoChallengeDieReorientP2-v0", **kwargs)
        elif env_name == "MyoBaodingBallsP2":
            return gym.make("myoChallengeBaodingP2-v1", **kwargs)
        elif env_name == "CustomMyoBaodingBallsP2":
            return gym.make("CustomMyoChallengeBaodingP2-v1", **kwargs)
        elif env_name == "MixtureModelBaodingEnv":
            return gym.make("MixtureModelBaoding-v1", **kwargs)
        elif env_name == "CustomMyoElbowPoseFixed":
            return gym.make("CustomMyoElbowPoseFixed-v0", **kwargs)
        elif env_name == "CustomMyoElbowPoseRandom":
            return gym.make("CustomMyoElbowPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoFingerPoseFixed":
            return gym.make("CustomMyoFingerPoseFixed-v0", **kwargs)
        elif env_name == "CustomMyoFingerPoseRandom":
            return gym.make("CustomMyoFingerPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoHandPoseFixed":
            return gym.make("CustomMyoHandPoseFixed-v0", **kwargs)
        elif env_name == "CustomMyoHandPoseRandom":
            return gym.make("CustomMyoHandPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoPenTwirlRandom":
            return gym.make("CustomMyoHandPenTwirlRandom-v0", )
        else:
            raise ValueError("Environment name not recognized:", env_name)
