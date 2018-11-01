#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import TensorDataset, DataLoader


from mpc import mpc
from mpc.mpc import GradMethods, QuadCost
from mpc.dynamics import NNDynamics
import mpc.util as eutil
from mpc.env_dx import pendulum, cartpole

import numpy as np
import numpy.random as npr

import argparse
import os
import sys
import shutil
import time
import re

import pickle as pkl

from setproctitle import setproctitle

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/pendulum.pkl')
    parser.add_argument('--work', type=str, default='./work')
    parser.add_argument('--save', type=str)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--mode', type=str, default='nn',
                        choices=['nn', 'empc', 'sysid'])
    parser.add_argument('--learn_cost', action='store_true')
    parser.add_argument('--learn_dx', action='store_true')
    # parser.add_argument('--mpc-pretrained', type=str, default='TODO')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--n_train', type=int, default=100)

    args = parser.parse_args()
    args.device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    if args.mode == 'empc':
        assert args.learn_cost or args.learn_dx

    if args.mode == 'sysid':
        args.learn_dx = True

    exp = IL_Exp(**vars(args))
    exp.run()


class IL_Exp:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

        torch.manual_seed(self.seed)

        self.env_name = re.search('\/([^\/]*)\.pkl', self.data).group(1)
        with open(self.data, 'rb') as f:
            self.env = pkl.load(f)

        tag = 'il.{}.{}.n_train={}'.format(self.env_name, self.mode, self.n_train)
        if self.learn_cost:
            tag += '.learn_cost'
        if self.learn_dx:
            tag += '.learn_dx'
        setproctitle('bamos.'+tag+'.{}'.format(self.seed))

        self.restart_warmstart_every = 50

        if not self.save:
            self.save = os.path.join(self.work, tag, str(self.seed))

        n_state, n_ctrl = self.env.true_dx.n_state, self.env.true_dx.n_ctrl
        self.n_state, self.n_ctrl = n_state, n_ctrl
        T = self.env.mpc_T
        self.T = T
        n_sc = n_state + n_ctrl

        if self.mode == 'nn':
            n_hidden = 256
            self.state_emb = nn.Sequential(
                nn.Linear(n_state, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
            ).to(self.device)
            self.ctrl_emb = nn.Sequential(
                nn.Linear(n_ctrl, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
            ).to(self.device)
            self.decode = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_ctrl),
            ).to(self.device)
            self.cell = nn.LSTMCell(n_hidden, n_hidden).to(self.device)
        # elif self.mode == 'mpc-dx':
        #     n_hidden = 64
        #     self.dx = NNDynamics(n_state, n_ctrl, hidden_sizes=[n_hidden, n_hidden])
        #     self.dx = self.dx.to(self.device)
        elif self.mode in ['empc', 'sysid']:
            self.true_q, self.true_p = self.env.true_dx.get_true_obj()

            self.learn_q_logit = torch.zeros_like(self.true_q, requires_grad=True)
            self.learn_p = torch.zeros_like(self.true_p, requires_grad=True)
            # self.learn_q = (self.true_q/(1-self.true_q+1e-8)).log().clone().requires_grad_()
            # self.learn_p = self.true_p.clone().requires_grad_()

            self.learn_q_logit = self.learn_q_logit.to(self.device)
            self.learn_p = self.learn_p.to(self.device)

            if self.learn_dx:
                if self.env_name == 'pendulum':
                    self.env_params = torch.tensor(
                        (15., 3., 0.5), requires_grad=True)
                elif self.env_name == 'cartpole':
                    self.env_params = torch.tensor(
                        (9.8, 3.0, 0.1, 1.0), requires_grad=True)
                elif self.env_name == 'pendulum-complex':
                    # self.env_params = torch.tensor(
                    #     (15., 3., 0.5), requires_grad=True)
                    torch.manual_seed(self.seed)
                    self.env_params = torch.tensor((5., 1., 1.)) + \
                        torch.tensor((3., 1., 1.))*(torch.rand(3)-0.5)
                    self.env_params.requires_grad_()

                    # n_hidden = 256
                    # self.extra_dx = NNDynamics(
                    #     n_state, n_ctrl, hidden_sizes=[n_hidden, n_hidden])
                else:
                    assert False
            else:
                self.env_params = self.env.true_dx.params
            self.env_params = self.env_params.to(self.device)
        else:
            assert False

        if os.path.exists(self.save):
            shutil.rmtree(self.save)
        os.makedirs(self.save)


    def lstm_forward(self, xinits):
        yt = self.state_emb(xinits)
        cell_state = None
        uts = []

        for t in range(self.T):
            cell_state = self.cell(yt, cell_state)
            ht, ct = cell_state
            ut = self.decode(ct)
            uts.append(ut)
            yt = self.ctrl_emb(ut)

        uts = torch.stack(uts, dim=1)
        return uts

    def run(self):
        torch.manual_seed(self.seed)

        loss_names = ['epoch']
        loss_names.append('im_loss')
        if self.learn_dx:
            loss_names.append('sysid_loss')
        fname = os.path.join(self.save, 'train_losses.csv')
        train_loss_f = open(fname, 'w')
        train_loss_f.write('{}\n'.format(','.join(loss_names)))
        train_loss_f.flush()

        fname = os.path.join(self.save, 'val_test_losses.csv')
        vt_loss_f = open(fname, 'w')
        loss_names = ['epoch']
        loss_names += ['im_loss_val', 'im_loss_test']
        # if self.learn_dx:
        #     loss_names += ['sysid_loss_val', 'im_loss_val']
        vt_loss_f.write('{}\n'.format(','.join(loss_names)))
        vt_loss_f.flush()

        if self.learn_dx:
            fname = os.path.join(self.save, 'dx_hist.csv')
            dx_f = open(fname, 'w')
            dx_f.write(','.join(map(str,
                self.env.true_dx.params.cpu().detach().numpy().tolist())))
            dx_f.write('\n')
            dx_f.flush()

        if self.learn_cost:
            fname = os.path.join(self.save, 'cost_hist.csv')
            cost_f = open(fname, 'w')
            cost_f.write(','.join(map(str,
                torch.cat((self.true_q, self.true_p))
                .cpu().detach().numpy().tolist())))
            cost_f.write('\n')
            cost_f.flush()

        if self.mode == 'nn':
            opt = optim.Adam(
                list(self.state_emb.parameters()) +
                list(self.ctrl_emb.parameters()) +
                list(self.decode.parameters()) +
                list(self.cell.parameters()),
                1e-4)
        elif self.mode == 'empc':
            params1 = []

            if self.learn_cost:
                params1 += [self.learn_q_logit, self.learn_p]
            if self.learn_dx:
                params1.append(self.env_params)


            params = [{
                'params': params1,
                'lr': 1e-2,
                'alpha': 0.5,
            }]

            # if self.learn_dx and self.env_name == 'pendulum-complex':
            #     params.append({
            #         'params': self.extra_dx.parameters(),
            #         'lr': 1e-4,
            #     })

            opt = optim.RMSprop(params)
        elif self.mode == 'sysid':
            params = [{
                'params': self.env_params,
                'lr': 1e-2,
                'alpha': 0.5,
            }]

            # if self.env_name == 'pendulum-complex':
                # params.append({
                #     'params': self.extra_dx.parameters(),
                #     'lr': 1e-4,
                # })
            opt = optim.RMSprop(params)
        else:
            assert False

        T = self.env.mpc_T

        if self.mode in ['empc', 'sysid']:
            train_warmstart = torch.zeros(self.n_train, T, self.n_ctrl).to(self.device)
            val_warmstart = torch.zeros(
                self.env.val_data.shape[0], T, self.n_ctrl).to(self.device)
            test_warmstart = torch.zeros(
                self.env.test_data.shape[0], T, self.n_ctrl).to(self.device)
        else:
            train_warmstart = val_warmstart = test_warmstart = None

        train_data, train = self.make_data(
            self.env.train_data[:self.n_train], shuffle=True)
        val_data, val = self.make_data(self.env.val_data)
        test_data, test = self.make_data(self.env.test_data)

        best_val_loss = None

        true_q, true_p = self.env.true_dx.get_true_obj()
        true_q, true_p = true_q.to(self.device), true_p.to(self.device)

        n_train_batch = len(train)
        # nom_u = None # TODO

        learn_cost_round_robin_interval = 10
        cost_update_q = False

        for i in range(self.n_epoch):
            if i > 0 and i % learn_cost_round_robin_interval == 0:
                cost_update_q = not cost_update_q

            if self.mode in ['empc', 'sysid'] \
               and i % self.restart_warmstart_every == 0:
                train_warmstart.zero_()
                val_warmstart.zero_()
                test_warmstart.zero_()

            for j, (xinits,xs,us,idxs) in enumerate(train):
                if self.mode == 'nn':
                    # pred_u = self.policy(xinits)
                    # pred_u = pred_u.reshape(-1, self.env.mpc_T, self.n_ctrl)
                    pred_u = self.lstm_forward(xinits)
                    assert pred_u.shape == us.shape
                    im_loss = (us.detach()-pred_u).pow(2).mean()
                elif self.mode in ['empc', 'sysid']:
                    if self.learn_dx:
                        if self.env_name == 'pendulum-complex':
                            dx = pendulum.PendulumDx(
                                self.env_params, simple=True)

                            # TODO: Hacky to have this here.
                            # class CombDx(nn.Module):
                            #     def __init__(self):
                            #         super().__init__()

                            #     def forward(_self, x, u):
                            #         return simple_dx(x,u) + 0.1*self.extra_dx(x,u)

                            # dx = CombDx()
                        else:
                            dx = self.env.true_dx.__class__(self.env_params)
                    else:
                        dx = self.env.true_dx

                    if self.learn_cost:
                        q = torch.sigmoid(self.learn_q_logit)
                        p = q.sqrt()*self.learn_p
                    else:
                        q, p = true_q, true_p

                    nom_x, nom_u = self.env.mpc(
                        dx, xinits, q, p,
                        u_init=train_warmstart[idxs].transpose(0,1),
                        # u_init=nom_u,
                        # eps_override=0.1,
                        # lqr_iter_override=100,
                    )
                    nom_u = nom_u.transpose(0,1)
                    train_warmstart[idxs] = nom_u
                    assert nom_u.shape == us.shape
                    im_loss = (us.detach()-nom_u).pow(2).mean()

                    if self.learn_dx:
                        xs_flat = xs[:,:-1].transpose(0,2).contiguous().view(
                            self.n_state, -1).t()
                        us_flat = us[:,:-1].transpose(0,2).contiguous().view(
                            self.n_ctrl, -1).t()
                        pred_next_x = dx(xs_flat, us_flat).t().view(
                            self.n_state, T-1, -1).transpose(0,2)
                        next_x = xs[:,1:]
                        assert next_x.shape == pred_next_x.shape
                        sysid_loss = (next_x.detach() - pred_next_x).pow(2).mean()
                else:
                    assert False


                t = [i+j/n_train_batch, im_loss.item()]
                if self.learn_dx:
                    t.append(sysid_loss.item())
                t = ','.join(map(str, t))
                print(t)
                train_loss_f.write(t+'\n')
                train_loss_f.flush()
                opt.zero_grad()
                if self.mode == 'sysid':
                    sysid_loss.backward()
                else:
                    im_loss.backward()

                if self.learn_cost:
                    if cost_update_q:
                        print('only updating q')
                        self.learn_p.grad.zero_()
                    else:
                        print('only updating p')
                        self.learn_q_logit.grad.zero_()

                if self.learn_dx:
                    if self.env_name == 'pendulum-complex':
                        true_params = self.env.true_dx.params[:3]
                    else:
                        true_params = self.env.true_dx.params
                    print(
                        np.array_str(
                            torch.stack((self.env_params, true_params))
                        .cpu().detach().numpy(), precision=2, suppress_small=True)
                    )
                    dx_f.write(','.join(map(str,
                        self.env_params.cpu().detach().numpy().tolist())))
                    dx_f.write('\n')
                    dx_f.flush()
                if self.learn_cost:
                    print(np.array_str(torch.stack((
                        torch.cat((true_q, true_p)),
                        torch.cat((q, p)),
                        # torch.cat((q.grad, p.grad)),
                    )).cpu().detach().numpy(), precision=2, suppress_small=True))
                    cost_f.write(','.join(map(str,
                        torch.cat((q, p))
                        .cpu().detach().numpy().tolist())))
                    cost_f.write('\n')
                    cost_f.flush()

                opt.step()
                # import ipdb; ipdb.set_trace()

                # if self.learn_cost:
                #     I = self.learn_q.data < 1e-6
                #     self.learn_q.data[I] = 1e-6

            val_loss = self.dataset_loss(val, val_warmstart)
            test_loss = self.dataset_loss(test, test_warmstart)
            t = [i, val_loss, test_loss]
            t = ','.join(map(str, t))
            vt_loss_f.write(t+'\n')
            vt_loss_f.flush()

            self.last_epoch = i
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                fname = os.path.join(self.save, 'best.pkl')
                print('Saving best model to {}'.format(fname))
                with open(fname, 'wb') as f:
                    pkl.dump(self, f)


    def make_data(self, data, warmstart=None, shuffle=False):
        data = data.to(self.device)
        xs, us = data[:,:,:self.n_state], data[:,:,-self.n_ctrl:]
        xinits = xs[:,0]
        n_data = xinits.shape[0]
        ds = TensorDataset(xinits, xs, us, torch.arange(0,n_data))
        loader = DataLoader(ds, batch_size=self.n_batch, shuffle=shuffle)
        return ds, loader


    def dataset_loss(self, loader, warmstart=None):
        true_q, true_p = self.env.true_dx.get_true_obj()
        true_q, true_p = true_q.to(self.device), true_p.to(self.device)

        losses = []
        for i, (xinits,xs,us,idxs) in enumerate(loader):
            n_batch = xinits.shape[0]

            if self.mode == 'nn':
                # pred_u = self.policy(xinits)
                # pred_u = pred_u.reshape(-1, self.env.mpc_T, self.n_ctrl)
                pred_u = self.lstm_forward(xinits)
            elif self.mode in ['empc', 'sysid']:
                if self.env_name == 'pendulum-complex':
                    if self.learn_dx:
                        dx = pendulum.PendulumDx(
                            self.env_params, simple=True)

                        # TODO: Hacky to have this here.
                        # class CombDx(nn.Module):
                        #     def __init__(self):
                        #         super().__init__()

                        #     def forward(_self, x, u):
                        #         return simple_dx(x,u) + 0.1*self.extra_dx(x,u)

                        # dx = CombDx()
                    else:
                        dx = pendulum.PendulumDx(
                            self.env_params, simple=False)

                    # TODO: Hacky to have this here.
                    # class CombDx(nn.Module):
                    #     def __init__(self):
                    #         super().__init__()

                    #     def forward(_self, x, u):
                    #         return simple_dx(x,u) + 0.1*self.extra_dx(x,u)

                    # dx = CombDx()
                else:
                    dx = self.env.true_dx.__class__(self.env_params)

                if self.learn_cost:
                    q = torch.sigmoid(self.learn_q_logit)
                    p = q.sqrt()*self.learn_p
                else:
                    q, p = true_q, true_p

                _, pred_u = self.env.mpc(
                    dx, xinits, q, p,
                    u_init=warmstart[idxs].transpose(0,1),
                    # lqr_iter_override=100,
                )
                pred_u = pred_u.transpose(0,1)
                warmstart[idxs] = pred_u

            assert pred_u.shape == us.shape
            loss = (us.detach()-pred_u).pow(2).mean(dim=1)
            losses.append(loss)

        loss = torch.cat(losses).mean().item()
        return loss


if __name__ == "__main__":
    main()
