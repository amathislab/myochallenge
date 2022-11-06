from definitions import ROOT_DIR
from models.classifier import collect_data_for_classifier, train_task_classifier

if __name__ == "__main__":
    # define paths
    model_path = ROOT_DIR + "/trained_models/baoding_phase2/alberto_518/best_model"
    env_path = ROOT_DIR + "/trained_models/baoding_phase2/alberto_518/training_env.pkl"
    save_path = ROOT_DIR + "/output/classifier/test_data.csv"

    # collect data
    collect_data_for_classifier(
        model_path,
        env_path,
        save_path,
        n_episodes=10,
    )

    # train classifier
    train_task_classifier(data_path=save_path)
