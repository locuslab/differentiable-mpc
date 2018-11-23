#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

import numpy as np
import numpy.random as npr

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import time
import os
import shutil
import pickle as pkl
import collections

import argparse
import setproctitle

# import setGPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_state', type=int, default=3)
    parser.add_argument('--n_ctrl', type=int, default=3)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '.'.join(["{}={}".format(x, getattr(args, x))
                  for x in ['n_state', 'n_ctrl', 'T']])
    setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work, t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = 'cuda' if args.cuda else 'cpu'

    n_state, n_ctrl = args.n_state, args.n_ctrl
    n_sc = n_state+n_ctrl

    expert_seed = 42
    assert expert_seed != args.seed
    torch.manual_seed(expert_seed)

    Q = torch.eye(n_sc)
    p = torch.randn(n_sc)

    alpha = 0.2

    expert = dict(
        Q = torch.eye(n_sc).to(device),
        p = torch.randn(n_sc).to(device),
        A = (torch.eye(n_state) + alpha*torch.randn(n_state, n_state)).to(device),
        B = torch.randn(n_state, n_ctrl).to(device),
    )
    fname = os.path.join(args.save, 'expert.pkl')
    with open(fname, 'wb') as f:
        pkl.dump(expert, f)

    torch.manual_seed(args.seed)
    A = (torch.eye(n_state) + alpha*torch.randn(n_state, n_state))\
        .to(device).requires_grad_()
    B = torch.randn(n_state, n_ctrl).to(device).requires_grad_()

    # u_lower, u_upper = -10., 10.
    u_lower, u_upper = None, None
    delta = u_init = None

    fname = os.path.join(args.save, 'losses.csv')
    loss_f = open(fname, 'w')
    loss_f.write('im_loss,mse\n')
    loss_f.flush()

    def get_loss(x_init, _A, _B):
        F = torch.cat((expert['A'], expert['B']), dim=1) \
            .unsqueeze(0).unsqueeze(0).repeat(args.T, n_batch, 1, 1)
        x_true, u_true, objs_true = mpc.MPC(
            n_state, n_ctrl, args.T,
            u_lower=u_lower, u_upper=u_upper, u_init=u_init,
            lqr_iter=100,
            verbose=-1,
            exit_unconverged=False,
            detach_unconverged=False,
            n_batch=n_batch,
        )(x_init, QuadCost(expert['Q'], expert['p']), LinDx(F))

        F = torch.cat((_A, _B), dim=1) \
            .unsqueeze(0).unsqueeze(0).repeat(args.T, n_batch, 1, 1)
        x_pred, u_pred, objs_pred = mpc.MPC(
            n_state, n_ctrl, args.T,
            u_lower=u_lower, u_upper=u_upper, u_init=u_init,
            lqr_iter=100,
            verbose=-1,
            exit_unconverged=False,
            detach_unconverged=False,
            n_batch=n_batch,
        )(x_init, QuadCost(expert['Q'], expert['p']), LinDx(F))

        traj_loss = torch.mean((u_true - u_pred)**2) + \
                    torch.mean((x_true - x_pred)**2)
        return traj_loss

    opt = optim.RMSprop((A, B), lr=1e-2)

    n_batch = 128
    for i in range(5000):
        x_init = torch.randn(n_batch,n_state).to(device)
        traj_loss = get_loss(x_init, A, B)

        opt.zero_grad()
        traj_loss.backward()
        opt.step()

        model_loss = torch.mean((A-expert['A'])**2) + \
                     torch.mean((B-expert['B'])**2)

        loss_f.write('{},{}\n'.format(traj_loss.item(), model_loss.item()))
        loss_f.flush()

        plot_interval = 100
        if i % plot_interval == 0:
            os.system('./plot.py "{}" &'.format(args.save))
            print(A, expert['A'])
        print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
            i, traj_loss.item(), model_loss.item()))

        # except KeyboardInterrupt: TODO
        #     raise
        # except Exception as e:
        #     # print(e)
        #     # pass
        #     raise




if __name__=='__main__':
    main()
