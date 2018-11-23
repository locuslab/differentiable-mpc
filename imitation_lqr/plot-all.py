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

    fig, ax = plt.subplots(figsize=(6,3))

    for seed in os.listdir(exp_dir):
        fname = os.path.join(exp_dir, seed, 'losses.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname)

            y = df['im_loss']
            y = np.convolve(y, np.full(N, 1./N), mode='valid')
            x = np.arange(len(y))+N

            ax.plot(x, y)

    ax.set_xlabel('Iteration')
    ax.set_ylim((0., None))
    ax.set_xlim((0., 1000.))
    ax.set_ylabel('Imitation Loss')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fname = os.path.join(SCRIPT_DIR, 'im_loss.{}'.format(ext))
        fig.savefig(fname)
        print('Saving to: {}'.format(fname))

    fig, ax = plt.subplots(figsize=(6,3))

    for seed in os.listdir(exp_dir):
        fname = os.path.join(exp_dir, seed, 'losses.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname)

            y = df['mse']
            y = np.convolve(y, np.full(N, 1./N), mode='valid')
            x = np.arange(len(y))+N

            ax.plot(x, y)

    ax.set_xlabel('Iteration')
    ax.set_ylim((0., 3.))
    ax.set_xlim((0., 1000.))
    # ax.set_xlim(N, 1000)
    # ax.set_xscale('log')
    # ax.set_ylim(1e-5, 1e1)
    ax.set_ylabel('Model Loss')
    # ax.set_yscale('log')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fname = os.path.join(SCRIPT_DIR, 'mse.{}'.format(ext))
        fig.savefig(fname)
        print('Saving to: {}'.format(fname))


if __name__ == "__main__":
    main()
