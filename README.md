# Differentiable MPC for End-to-end Planning and Control

This repository is by [Brandon Amos](http://bamos.github.io),
Ivan Dario Jimenez Rodriguez, Jacob Sacks, Byron Boots,
and [J. Zico Kolter](http://zicokolter.com)
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our NIPS 2018 paper
[Differentiable MPC for End-to-end Planning and Control](https://arxiv.org/abs/1810.13400).

The PyTorch implementation of the fast and differentiable MPC solver
we developed for this work is available as a standalone library at
[locuslab/mpc.pytorch](https://locuslab.github.io/mpc.pytorch/).

If you find this repository helpful in your publications,
please consider citing our paper.

```
@article{amos2018differentiable,
  title={{Differentiable MPC for End-to-end Planning and Control}},
  author={Brandon Amos and Ivan Jimenez and Jacob Sacks and Byron Boots and J. Zico Kolter},
  booktitle={{Advances in Neural Information Processing Systems}},
  year={2018}
}
```

## Setup and Dependencies

+ Python/numpy/[PyTorch](https://pytorch.org)
+ [locuslab/mpc.pytorch](https://github.com/locuslab/mpc.pytorch)

# LQR Imitation Learning Experiments

From within the `imitation_lqr` directory:
1. `train.py` is the main training script for the experiment 
   in Section 5.3.

# Non-Convex Imitation Learning Experiments

From within the `imitation_nonconvex` directory:
1. `make_dataset.py` should be run to create a dataset of trajectories
   for each environment.
2. `il_exp.py` is the main training script for each experiment.
3. `run-pendulum-cartpole.sh` runs all of the experiments for the
   pendulum and cartpole environments in Section 5.3.
3. `run-complex-pendulum.sh` runs all of the experiments for the
   non-realizable pendulum environment in Section 5.4.
