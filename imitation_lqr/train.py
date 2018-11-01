#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

import numpy as np
import numpy.random as npr

from mpc import mpc
from mpc.mpc import GradMethods
from mpc.dynamics import NNDynamics
import mpc.util as eutil
from mpc.env_dx import pendulum, cartpole

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import time
import os
import shutil
import json
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
    setproctitle.setproctitle('bamos.'+t+'.{}'.format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work, t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    meta_file = os.path.join(args.save, 'meta.json')
    meta = create_experiment(args.n_state, args.n_ctrl, args.T)
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=4)


    true_model = {}
    for k in ['Q', 'p', 'A', 'B']:
        v = torch.Tensor(np.array(meta[k])).double()
        if torch.cuda.is_available():
            v = v.cuda()
        v = Variable(v)
        meta[k] = v
        true_model[k] = v


    n_state, n_ctrl, alpha = args.n_state, args.n_ctrl, meta['alpha']
    npr.seed(1) # Intentionally 1 instead of args.seed so these are the same.
    A_model = np.eye(n_state) + alpha*np.random.randn(n_state, n_state)
    B_model = npr.randn(n_state, n_ctrl)
    dtype = true_model['Q'].data.type()
    A_model = Parameter(torch.Tensor(A_model).type(dtype))
    B_model = Parameter(torch.Tensor(B_model).type(dtype))


    # u_lower, u_upper = -100., 100.
    u_lower, u_upper = -1., 1.
    delta = u_init = None

    optimizer = optim.RMSprop((A_model, B_model), lr=1e-2)

    torch.manual_seed(args.seed)

    fname = os.path.join(args.save, 'losses.csv')
    loss_f = open(fname, 'w')
    loss_f.write('im_loss,mse\n')
    loss_f.flush()

    n_batch = 64
    for i in range(5000):
      x_init = Variable(1.*torch.randn(n_batch,n_state).type(dtype))
      optimizer.zero_grad()

      try:
          F = torch.cat((true_model['A'], true_model['B']), dim=1) \
              .unsqueeze(0).unsqueeze(0).repeat(args.T, n_batch, 1, 1)
          x_true, u_true, objs_true = mpc.MPC(
              n_state, n_ctrl, args.T, x_init,
              u_lower=u_lower, u_upper=u_upper, u_init=u_init,
              mpc_iter=100,
              verbose=-1,
              exit_unconverged=False,
              detach_unconverged=False,
              F=F,
              n_batch=n_batch,
          )(true_model['Q'], true_model['p'])

          F = torch.cat((A_model, B_model), dim=1) \
              .unsqueeze(0).unsqueeze(0).repeat(args.T, n_batch, 1, 1)
          x_pred, u_pred, objs_pred = mpc.MPC(
              n_state, n_ctrl, args.T, x_init,
              u_lower=u_lower, u_upper=u_upper, u_init=u_init,
              mpc_iter=100,
              verbose=-1,
              exit_unconverged=False,
              detach_unconverged=False,
              F=F,
              n_batch=n_batch,
          )(true_model['Q'], true_model['p'])

          traj_loss = torch.mean((u_true - u_pred)**2)
                      # torch.mean((x_true-x_pred)**2)
          traj_loss.backward()
          optimizer.step()
          # import ipdb; ipdb.set_trace()

          model_loss = torch.mean((A_model-true_model['A'])**2) + \
                       torch.mean((B_model-true_model['B'])**2)

          loss_f.write('{},{}\n'.format(traj_loss.data[0], model_loss.data[0]))
          loss_f.flush()

          plot_interval = 100
          if i % plot_interval == 0:
              os.system('./plot.py "{}" &'.format(args.save))
              print(A_model,true_model['A'])
          print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
              i, traj_loss.data[0], model_loss.data[0]))
      except KeyboardInterrupt:
          raise
      except Exception as e:
          # print(e)
          # pass
          raise



def create_experiment(n_state, n_ctrl, T):
    n_sc = n_state+n_ctrl
    npr.seed(2)
    Q = np.eye(n_sc)
    Q = np.tile(Q, (T, 1, 1))
    p = npr.randn(T, n_sc)

    alpha = 0.2
    A = np.eye(n_state) + alpha*np.random.randn(n_state, n_state)
    B = npr.randn(n_state, n_ctrl)

    meta = dict(
        Q = Q.tolist(),
        p = p.tolist(),
        alpha = alpha,
        A = A.tolist(),
        B = B.tolist(),
    )
    return meta


if __name__=='__main__':
    main()
