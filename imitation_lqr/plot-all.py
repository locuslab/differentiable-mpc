#!/usr/bin/env python3

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

# from matplotlib import rc
# rc('text', usetex=True)

import seaborn as sns

import numpy as np

import pandas as pd

from collections import namedtuple

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    N = 25

    exp_dir = os.path.join(SCRIPT_DIR, 'work', 'n_state=3.n_ctrl=3.T=5')

    im_data, mse_data = [], []
    for seed in os.listdir(exp_dir):
        fname = os.path.join(exp_dir, seed, 'losses.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname)

            y = df['im_loss']
            y = np.convolve(y, np.full(N, 1./N), mode='valid')
            im_data.append(y)

            y = df['mse']
            y = np.convolve(y, np.full(N, 1./N), mode='valid')
            mse_data.append(y)

    min_len = min(map(len, im_data))
    im_data = np.stack([d[:min_len] for d in im_data])
    mse_data = np.stack([d[:min_len] for d in mse_data])

    fig, ax = plt.subplots(figsize=(6,3))
    mean = im_data.mean(axis=0)
    std = im_data.std(axis=0)
    x = np.arange(len(mean))+N
    l, = ax.plot(x, mean)
    ax.fill_between(x, mean-std, mean+std, color=l.get_color(), alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_xlim(N, 1000)
    ax.set_xscale('log')
    ax.set_ylim(1e-4, 1e0)
    ax.set_ylabel('Imitation Loss')
    ax.set_yscale('log')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fname = os.path.join(SCRIPT_DIR, 'im_loss.{}'.format(ext))
        fig.savefig(fname)
        print('Saving to: {}'.format(fname))

    fig, ax = plt.subplots(figsize=(6,3))
    mean = mse_data.mean(axis=0)
    std = mse_data.std(axis=0)
    x = np.arange(len(mean))+N
    l, = ax.plot(x, mean)
    ax.fill_between(x, mean-std, mean+std, color=l.get_color(), alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_xlim(N, 1000)
    ax.set_xscale('log')
    ax.set_ylim(1e-5, 1e1)
    ax.set_ylabel('Model Loss')
    ax.set_yscale('log')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fname = os.path.join(SCRIPT_DIR, 'mse.{}'.format(ext))
        fig.savefig(fname)
        print('Saving to: {}'.format(fname))


if __name__ == "__main__":
    main()
