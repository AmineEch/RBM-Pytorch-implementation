import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.distributions.bernoulli import Bernoulli as B

class RBM(nn.Module):
    '''
    class implementing the restricted boltzmann machine model
    '''

    def __init__(self, n_v, n_h):
        '''
        self.params[0] = W, weights
        self.params[1] = c, v bias
        self.params[2] = b, h bias
        '''
        super(RBM, self).__init__()
        self.params = [nn.Parameter(torch.randn(n_h, n_v)), nn.Parameter(torch.zeros(n_v)),
                       nn.Parameter(torch.zeros(n_h))]
        self.n_v = n_v
        self.n_h = n_h

    def forward(self, x, x_t, h_t):
        '''
        Forward function : p(x) = exp(-F(x)) F: Free energy
        '''
        F = - self.params[1].matmul(x) - nn.Softplus()(self.params[0].matmul(x).add_(self.params[2])).sum()
        E = - self.params[1].matmul(x_t) - self.params[2].matmul(h_t) - h_t.matmul(self.params[0].matmul(x_t))
        loss = F - E + 0.0005 * (Func.mse_loss(self.params[0], torch.zeros_like(self.params[0]))
                                 + Func.mse_loss(self.params[1], torch.zeros_like(self.params[1]))
                                 + Func.mse_loss(self.params[2], torch.zeros_like(self.params[2])) )
        return loss

    def sample_h_knowing_x(self, x):
        '''

        :param x:
        :return: h
        '''
        probs = Func.sigmoid(self.params[2] + self.params[0].matmul(x))
        h = B(probs).sample()
        return h

    def sample_x_knowing_h(self, h):
        '''
        :param h:
        :return: x
        '''
        probs = Func.sigmoid(self.params[1] + h.matmul(self.params[0]))
        x = B(probs).sample()
        return x

    def gibbs_sampling(self, num_itter, x_init, h_init):
        '''
        :param num_itter:
        :param x_init:
        :param h_init:
        :return: (x, h)
        '''

        for i in range(num_itter+1):
            if i == 0 :
                x_t, h_t = x_init, h_init
            else:
                h_t = self.sample_h_knowing_x(x_t)
                x_t = self.sample_x_knowing_h(h_t)

        return x_t, h_t