#!/usr/bin/env python3

import argparse
import pickle as pkl
import os
import sys

from il_env import IL_Env


from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--env_name', type=str, default='pendulum')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)

    n_train, n_val, n_test = 100, 10, 10
    env = IL_Env(args.env_name, lqr_iter=500)
    env.populate_data(n_train=n_train, n_val=n_val, n_test=n_test, seed=0)

    save = os.path.join(args.data_dir, args.env_name+'.pkl')
    print('Saving data to {}'.format(save))
    with open(save, 'wb') as f:
        pkl.dump(env, f)

if __name__ == "__main__":
    main()
