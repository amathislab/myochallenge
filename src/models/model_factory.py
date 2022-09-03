from ray.rllib.models import ModelCatalog


class ModelFactory:
    """Static factory to register models by name
    Raises:
        ValueError: when the input string does not correspond to any known model
    """

    @staticmethod
    def register(model_name):
        if model_name == "oracle_policy":
            from src.models.rma import OraclePolicyModel

            ModelCatalog.register_custom_model(model_name, OraclePolicyModel)
        elif model_name == "oracle_q":
            from src.models.rma import OracleQModel

            ModelCatalog.register_custom_model(model_name, OracleQModel)
        elif model_name == "tcn_policy":
            from src.models.rma import TCNPolicyModel

            ModelCatalog.register_custom_model(model_name, TCNPolicyModel)
        elif model_name == "oracle_q_adapt":
            from src.models.rma import OracleQAdaptModel

            ModelCatalog.register_custom_model(model_name, OracleQAdaptModel)
        elif model_name == "tcn_q":
            from src.models.rma import TCNQModel

            ModelCatalog.register_custom_model(model_name, TCNQModel)
        elif model_name == "dmap_policy":
            from src.models.dmap import DMAPPolicyModel

            ModelCatalog.register_custom_model(model_name, DMAPPolicyModel)
        elif model_name == "local_dmap_policy":
            from src.models.dmap import LocalDMAPPolicyModel

            ModelCatalog.register_custom_model(model_name, LocalDMAPPolicyModel)
        elif model_name == "sensory_target_dmap_policy":
            from src.models.dmap import SensoryTargetDMAPPolicyModel
            
            ModelCatalog.register_custom_model(model_name, SensoryTargetDMAPPolicyModel)
        elif model_name == "decision_dmap_policy":
            from src.models.dmap import DecisionDMAPPolicyModel
            
            ModelCatalog.register_custom_model(model_name, DecisionDMAPPolicyModel)           
        elif model_name == "dmap_q":
            from src.models.dmap import DMAPQModel

            ModelCatalog.register_custom_model(model_name, DMAPQModel)
        elif model_name == "simple_q":
            from src.models.rma import SimpleQModel

            ModelCatalog.register_custom_model(model_name, SimpleQModel)
        elif model_name == "sensory_target_simple_q":
            from src.models.rma import SensoryTargetSimpleQModel
            
            ModelCatalog.register_custom_model(model_name, SensoryTargetSimpleQModel)
        elif model_name == "decision_simple_q":
            from src.models.rma import DecisionSimpleQModel
            
            ModelCatalog.register_custom_model(model_name, DecisionSimpleQModel)
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