# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *

import torch
import math
from torch.autograd import Variable
import numpy as np

from utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
from density_plot import get_esd_plot


def KL_div(mu, logvar, reduction='none'):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))

    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return KL


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model_e, model_g, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model_e = model_e.eval()  # make model is in evaluation model
        self.model_g = model_g.eval()  # make model is in evaluation model
        self.model   = model_g.eval()
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda:1'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            z, sdj = self.model_e(self.inputs)
            loss = self.criterion(z, sdj)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model_e)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader
        count = 0
        THv = [torch.zeros(p.size()).to(device) for p in self.params
               ]  # accumulate result
        for inputs, targets in self.data:
            count += 1
            print(count)
            inputs = inputs.to(device)
            target = Variable(inputs.data.view(-1) * 255).long()
            self.model_e.zero_grad()
            tmp_num_data = inputs.size(0)

            [z, mu, logvar] = self.model_e(inputs)
            recon = self.model_g(z)
            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = self.criterion(recon, target)

            recl = torch.sum(recl) / tmp_num_data
            kld = KL_div(mu, logvar)

            loss = recl + kld.mean()
            loss.backward(create_graph=True)

            params, gradsH = get_params_grad(self.model_e)
            self.model_e.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)
            if count > 100:
                break
        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                 ]  # generate random vector
            v = normalization(v)  # normalize the vector
            print(computed_dim)
            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model_e.zero_grad()
                print('ith')
                print(i)
                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=500, n_v=10):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                print(i)
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = (b_[0, :] ** 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full

