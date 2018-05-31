import numpy as np
import torch
dtype = torch.DoubleTensor

from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import rbf_kernel
from torch.autograd import Variable

from utils import print_progress


class Kernel(ABC):

    @abstractmethod
    def __str__(self):
        """ Pretty-print the kernel """
        raise NotImplementedError('Kernel.show is an abstract method')

    @abstractmethod
    def get_kernel_matrix(self, X):
        """ Compute the kernel matrix K_xx = {k(x1, x2)} """
        raise NotImplementedError('Kernel.get_kernel_matrix is an abstract method')

    @abstractmethod
    def get_cross_kernel_matrix(self, X, Y):
        """ Compute the cross kernel matrix K_xy = {k(x, y)} """
        raise NotImplementedError('Kernel.get_cross_kernel_matrix is an abstract method')

    @abstractmethod
    def get_random_features(self, num_features):
        """ Construct a random features function """
        raise NotImplementedError('Kernel.get_random_features is an abstract method')

    def get_kernel_matrix_mean(self, X):
        """ Compute the means of the kernel matrix K_xx = {k(x1, x2)} """
        N = len(X)
        if N <= 10000:
            return np.mean(self.get_kernel_matrix(X))
        else:
            s = 0.0
            step = 10000 ** 2 // N
            start = 0
            while start < N:
                s += np.sum(self.get_cross_kernel_matrix(X[start:(start+step)], X))
                start = min(start + step, N)
                print_progress('Computing mean of K_xx... %.1f%%' % (100.0 * start / N))
            print('')
            return s / (N ** 2)

    def get_kernel_matrix_rowmeans(self, X, Y):
        """ Compute the row means of the cross kernel matrix K_xy = {k(x, y)} """
        M, N = len(X), len(Y)
        if N * M <= 10000 ** 2:
            return np.mean(self.get_cross_kernel_matrix(X, Y), axis=1)
        else:
            sums = np.zeros(M)
            step = 10000 ** 2 // M
            start = 0
            while start < N:
                sums += np.sum(self.get_cross_kernel_matrix(X, Y[start:(start+step)]), axis=1)
                start = min(start + step, N)
                print_progress('[Progress] Computing kernel matrix row means... %.1f%%' % (100.0 * start / N))
            print('')
            return sums / N


class EQKernel(Kernel):
    def __init__(self, D, gamma):
        self.D = D # number of input dimensions
        self.gamma = gamma # kernel parameter (roughly inverse lengthscale)

    def get_kernel_matrix(self, X):
        return rbf_kernel(X, gamma=self.gamma)

    def get_cross_kernel_matrix(self, X, Y):
        return rbf_kernel(X, Y, gamma=self.gamma)

    def get_random_features(self, J):
        w = np.random.randn(self.D, J) * np.sqrt(2.0 * self.gamma)
        b = 2 * np.pi * np.random.rand(J)
        def rf_np(X):
            return np.sqrt(2.0 / J) * np.cos(np.dot(X, w) + b)

        wTorch = Variable(torch.from_numpy(w), requires_grad=False)
        bTorch = Variable(torch.from_numpy(b), requires_grad=False)
        
        def rf_torch(X):
            return torch.cos(torch.mm(X, wTorch) + bTorch) * np.sqrt(2.0 / J)
        return rf_np, rf_torch

    def __str__(self):
        return 'EQKernel(D=%d, gamma=%f)' % (self.D, self.gamma)


# --- UTILITIES ---
def compute_mean_feature_vector(X, rf, J):
    N, D = np.shape(X)
    if N * J <= 10000 ** 2:
        return np.mean(rf(X), axis=0)
    else:
        sums = np.zeros(J)
        step = 10000 ** 2 // J
        start = 0
        while start < N:
            sums += np.sum(rf(X[start:(start+step)]), axis=0)
            start = min(start + step, N)
            print_progress('[Progress] Computing mean feature vector... done %.1f%%' % (100.0 * start / N))
        print('')
        return sums / N
