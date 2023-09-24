# MyoChallenge 2022

This is the code repository for winning the Boading ball task of the [MyoChallenge 2022](https://sites.google.com/view/myochallenge). Our team was named _stiff fingers_, and this name was inspired by the fact as the fingers were not too agile.... perhaps fingers are not task relevant.

Our team comprised:
- Alberto Chiappa and Alexander Mathis EPFL, Switzerland
- Pablo Tano, Nisheet Patel, Alexandre Pouget University of Geneva, Switzerland

Here we have documented a summary of our approach including our key insights and intuitions along with all the training steps and hyperparameters [here](docs/summary.md).

## Software requirements

Listed in `requirements.txt`. Note that there is a version error with some packages, e.g. `stable_baselines3`, requiring later versions of `gym` which `myosuite` is incompatible with. If your package manager automatically updates gym, do a `pip install gym==0.13.0` (or equivanlent with your package manager) at the end and this should work fine. If you experience any issues, feel free to open an issue on this repository or contact us via email.

## Usage

Run `python src/main_baoding.py` to start a training. Note that this starts training from one of the pre-trained models in our curriculum. You can find all the trained models along with the scripts used to train them and the environment configurations [here](trained_models). The full information about the training process can be found in the [summary](docs/summary.md).

To evaluate the best single policy network (see the [summary](docs/summary.md)), run `python src/main_eval.py`. To evaluate the final ensemble (55% score), run `python src/eval_mixture_of_ensembles.py`.

## Further context and literature

If you want to read more about the challenge and our solution, check out this exciting article in the [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v220/caggiano22a.html)! 

If you use our code, or ideas please cite:

```
@InProceedings{pmlr-v220-caggiano22a,
  title = 	 {MyoChallenge 2022: Learning contact-rich manipulation using a musculoskeletal hand},
  author =       {Caggiano, Vittorio and Durandau, Guillaume and Wang, Huwawei and Chiappa, Alberto and Mathis, Alexander and Tano, Pablo and Patel, Nisheet and Pouget, Alexandre and Schumacher, Pierre and Martius, Georg and Haeufle, Daniel and Geng, Yiran and An, Boshi and Zhong, Yifan and Ji, Jiaming and Chen, Yuanpei and Dong, Hao and Yang, Yaodong and Siripurapu, Rahul and Ferro Diez, Luis Eduardo and Kopp, Michael and Patil, Vihang and Hochreiter, Sepp and Tassa, Yuval and Merel, Josh and Schultheis, Randy and Song, Seungmoon and Sartori, Massimo and Kumar, Vikash},
  booktitle = 	 {Proceedings of the NeurIPS 2022 Competitions Track},
  pages = 	 {233--250},
  year = 	 {2022},
  editor = 	 {Ciccone, Marco and Stolovitzky, Gustavo and Albrecht, Jacob},
  volume = 	 {220},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28 Nov--09 Dec},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v220/caggiano22a/caggiano22a.pdf},
  url = 	 {https://proceedings.mlr.press/v220/caggiano22a.html},
  abstract = 	 {Manual dexterity has been considered one of the critical components for human evolution. The ability to perform movements as simple as holding and rotating an object in the hand without dropping it needs the coordination of more than 35 muscles which act synergistically or antagonistically on multiple joints. This complexity in control is markedly different from typical pre-specified movements or torque based controls used in robotics. In the MyoChallenge at the NeurIPS 2022 competition track, we challenged the community to develop controllers for a realistic hand to solve a series of dexterous manipulation tasks. The MyoSuite framework was used to train and test controllers on realistic, contact rich and computation efficient virtual neuromusculoskeletal model of the hand and wrist. Two tasks were proposed: a die re-orientation and a boading ball (rotation of two spheres respect to each other) tasks. More than 40 teams participated to the challenge and submitted more than 340 solutions. The challenge was split in two phases. In the first phase, where a limited set of objectives and randomization were proposed, teams managed to achieve high performance, in particular in the boading-ball task.  In the second phase as the focus shifted towards generalization of task solutions to extensive variations of object and task properties, teams saw significant performance drop. This shows that there is still a large gap in developing agents capable of generalizable skilled manipulation. In future challenges, we will continue pursuing the generalizability both in skills and agility of the tasks exploring additional realistic neuromusculoskeletal models.}
}
```

