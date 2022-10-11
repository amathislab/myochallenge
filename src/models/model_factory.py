from ray.rllib.models import ModelCatalog


class ModelFactory:
    """Static factory to register models by name
    Raises:
        ValueError: when the input string does not correspond to any known model
    """

    @staticmethod
    def register(model_name):
        if model_name == "dmap_policy":
            from src.models.dmap import DMAPPolicyModel

            ModelCatalog.register_custom_model(model_name, DMAPPolicyModel)
        elif model_name == "local_dmap_policy":
            from src.models.dmap import LocalDMAPPolicyModel

            ModelCatalog.register_custom_model(model_name, LocalDMAPPolicyModel)
        elif model_name == "nondmap_policy":
            from src.models.dmap import NonDMAPPolicyModel
            
            ModelCatalog.register_custom_model(model_name, NonDMAPPolicyModel)      
        elif model_name == "dmap_q":
            from src.models.dmap import DMAPQModel

            ModelCatalog.register_custom_model(model_name, DMAPQModel)
        elif model_name == "simple_q":
            from src.models.dmap import SimpleQModel

            ModelCatalog.register_custom_model(model_name, SimpleQModel)
        else:
            raise ValueError("Unknown model name", model_name)

    @staticmethod
    def register_models_from_config(policy_configs):
        for policy in policy_configs.values():
            for model_params in policy.values():
                if isinstance(model_params, dict):
                    model_name = model_params.get("custom_model")
                    if model_name is not None:
                        ModelFactory.register(model_name)