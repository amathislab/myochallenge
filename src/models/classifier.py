import datetime as dt
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from torch.utils.data import DataLoader, Dataset

from envs.environment_factory import EnvironmentFactory

# define constants
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
N_EPOCHS = 40
BATCH_SIZE = 100
N_OBS_PER_TRIAL = 13
DIMS_PER_OBS = 18


def get_config() -> dict:
    return {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "goal_time_period": (4, 6),
        "task_choice": "random",
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
    }


def load_normalized_envs(env_path: str, config: dict) -> VecNormalize:
    env = EnvironmentFactory.register("CustomMyoBaodingBallsP2", **config)
    envs = DummyVecEnv([lambda: env] * 16)

    return VecNormalize.load(env_path, envs)


def load_model_and_env(model_path: str, env_path: str):
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    envs = load_normalized_envs(env_path, config=get_config())
    model = RecurrentPPO.load(model_path, env=envs, custom_objects=custom_objects)

    return model, envs


@dataclass
class DataCollector:
    model_path: str
    env_path: str
    timestep = 0
    n_obs_per_trial = 50

    def __post_init__(self):
        self.model, self.env = load_model_and_env(self.model_path, self.env_path)
        self.obs_trial = []
        self.all_obs = []
        self.task_ids = []

    def collect_single_obs(self, obs) -> None:
        self.obs_trial.append(obs[29:47].copy())

    def concatenate_trial_obs(self, task_id) -> None:
        self.all_obs.append(np.concatenate(self.obs_trial))
        self.task_ids.append(task_id)

    def predict(self, obs, state, episode_start, deterministic):
        self.timestep += 1
        obs = self.env.normalize_obs(obs)

        return self.model.predict(obs, state, episode_start, deterministic)

    def collect_data(self, env, n_episodes: int = 10_000) -> None:
        for ep in range(n_episodes):
            self.obs_trial = []
            lstm_states = None
            obs = env.reset()
            episode_starts = np.ones((1,), dtype=bool)

            if ep % 100 == 0:
                print(f"Episode {ep}")

            for _ in range(self.n_obs_per_trial):
                self.collect_single_obs(obs)

                action, lstm_states = self.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=False,
                )
                obs, _, dones, _ = env.step(action)

                episode_starts = dones

            self.concatenate_trial_obs(env.which_task.value)

    def save_data(self, path: Path) -> None:
        df = pd.DataFrame(np.vstack(self.all_obs))
        df["task_id"] = self.task_ids
        df.to_csv(path, index=False)


def collect_data_for_classifier(
    model_path: str, env_path: str, save_path: str, n_episodes: int = 10_000
) -> None:
    env = EnvironmentFactory.register("CustomMyoBaodingBallsP2", **get_config())

    print("\n\nCollecting data\n")
    start = time.time()
    data_collector = DataCollector(model_path, env_path)
    data_collector.collect_data(env, n_episodes=n_episodes)
    data_collector.save_data(Path(save_path))
    print(f"Data collection took {dt.timedelta(seconds = (time.time() - start))}")


@dataclass
class TrainData(Dataset):
    X_data: torch.Tensor
    y_data: torch.Tensor

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


@dataclass
class TestData(Dataset):
    X_data: torch.Tensor

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class TaskClassifier(torch.nn.Module):
    def __init__(self, n_obs_per_trial=N_OBS_PER_TRIAL):
        super().__init__()

        self.layer_1 = torch.nn.Linear(n_obs_per_trial * DIMS_PER_OBS, 200)
        self.layer_2 = torch.nn.Linear(200, 100)
        self.layer_out = torch.nn.Linear(100, 1)

        # self.dropout = torch.nn.Dropout(p=0.1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        return self.layer_out(x)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train_task_classifier(
    data_path: str = "../output/classifier/data_for_baoding_task_classifier_alberto-518.csv",
) -> None:
    # read and preprocess data
    df = pd.read_csv(data_path)
    X = df.iloc[:, 0 : (N_OBS_PER_TRIAL * DIMS_PER_OBS)]
    y = df.iloc[:, -1].copy()
    y = y.clip(0, 1)  # only classify between task hold and rotate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=69
    )
    print("Fitting and scaling data")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # convert to tensors for torch
    train_data = TrainData(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train.to_numpy())
    )
    test_data = TestData(torch.FloatTensor(X_test))

    # use the DataLoader class to create an iterator for the data
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    # instantiate the model, loss, and optimizer
    task_classifier = TaskClassifier()
    task_classifier.to(device)
    print(task_classifier)

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )  # CrossEntropyLoss()   # use for binary classification
    optimizer = torch.optim.Adam(task_classifier.parameters(), lr=LEARNING_RATE)

    # train the model
    task_classifier.train()
    for e in range(1, N_EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = task_classifier(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # printing metrics
        loss = epoch_loss / len(train_loader)
        accuracy = epoch_acc / len(train_loader)
        print(f"Epoch {e+0:03}: | Loss: {loss:.5f} | Accuracy: {accuracy:.3f}")

    # evaluate the model
    y_pred_list = []
    task_classifier.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = task_classifier(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    # print results
    confusion_matrix(y_test, y_pred_list)
    print(classification_report(y_test, y_pred_list))

    # save task_classifier and data preprocessor
    save_folder = "../output/classifier/"
    with open(save_folder + "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    torch.save(task_classifier.state_dict(), save_folder + "task_classifier.pt")


if __name__ == "__main__":
    pass
    # print("Collecting data")
    # collect_data_for_classifier()

    # print("Training task classifier")
    # train_task_classifier()
