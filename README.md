# MyoChallenge

Code repository for [MyoChallenge](https://sites.google.com/view/myochallenge) by team _stiff fingers_. You can find a summary of our approach [here](docs/summary.md).

## Requirements

Listed in `requirements.txt`. Note that there is a version error with some packages, e.g. `stable_baselines3`, requiring later versions of `gym` which `myosuite` is incompatible with. If your package manager automatically updates gym, do a `pip install gym==0.13.0` (or equivanlent with your package manager) at the end and this should work fine. If you experience any issues, feel free to open an issue on this repository or contact us via email.

## Usage

Run `python src/main_baoding.py`. Note that this starts training from one of the pre-trained models in our curriculum. You can find all the trained models along with the scripts used to train them and the environment configurations [here](trained_models/baoding_winning_models/).
