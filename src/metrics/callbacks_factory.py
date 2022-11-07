from ray.rllib.algorithms.callbacks import DefaultCallbacks

from metrics.callbacks import FullInfoCallbacks, MyoInfoCallbacks


class CallbacksFactory:
    """Static factory to get a callback class by name. Returns the default
    callbacks class if it receives None.

    Raises:
        ValueError: when the input string does not correspond to a know callback.
    """

    @staticmethod
    def get_callbacks_class(name):
        if name is None:
            return DefaultCallbacks
        elif name == "FullInfoCallbacks":
            return FullInfoCallbacks
        elif name == "MyoInfoCallbacks":
            return MyoInfoCallbacks
        else:
            raise ValueError("Unknown callbacks name: ", name)
