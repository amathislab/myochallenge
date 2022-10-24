import gym
from ray.tune import register_env


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
        elif env_name == "MyoBaodingBallsP2":
            return gym.make("myoChallengeBaodingP2-v1", **kwargs)
        elif env_name == "CustomMyoBaodingBallsP2":
            return gym.make("CustomMyoChallengeBaodingP2-v1", **kwargs)
                   
    @staticmethod
    def register(env_name, **kwargs):
        """Registers the specified environment, so that it can be instantiated
        by the RLLib algorithms by name.

        Args:
            env_name (str): name of the environment

        Returns:
            gym.env: the registered environment
        """
        env = EnvironmentFactory.create(env_name, **kwargs)
        register_env(env_name, lambda _: EnvironmentFactory.create(env_name, **kwargs))
        return env
