# system imports
import io
import time
import os
from os.path import abspath, join, dirname, basename
import sys
import cProfile
import pstats
import itertools as it


# third party imports
import scipy
import numpy as np
import matplotlib.pyplot as plt
import parse  # used for loading data files
import pandas as pd
import opt_einsum as oe

class MPS_decompose(object):
    """
    The class test the algorithm that decompose an arbitrary state tensor into MPS
    """
    def __init__(self, num_site, num_state):
        """
        initialize input tensor
        num_site: number lattice sites
        num_state: number states at each local site
        """
        self.L = num_site
        self.D = num_state

        # initialize a random tensor with shape [D, D, ..., D] (L sites)

        # form the shape of the random tensor
        shape = []
        for i in range(self.L):
            shape.append(self.D)

        # pass the shape into rand function
        self.input_tensor = np.random.rand(*shape)

        # normalize the state vector <psi|psi> = 1
        norm = self.cal_norm(self.input_tensor)
        print("norm of input tensor the original tensor:{:}".format(norm))
        self.input_tensor /= np.sqrt(norm)

        # check the tensor is normalized
        assert np.allclose(1, self.cal_norm(self.input_tensor))

        # input tensor
        print("Shape of input tensor:{:}".format(self.input_tensor.shape))
        print("Input tensor:\n{:}".format(self.input_tensor))

    def cal_norm(self, input_tensor):
        """calcuate the 2-norm of an input tensor"""
        norm = 0
        for element in input_tensor.ravel():
            norm += element**2
        return norm


    def left_decompose(self):
        """algorithm to decompose the tensor to MPS from the left"""
        L, D = self.L, self.D

        # define an empty list to store MPS tensor from left to the right
        MPS_tensor = []

        # sweep from left to right to construct MPS tensors
        PSI = self.input_tensor.copy()
        # store bond dimension data
        r = [] # initial bond dimension
        for site in range(L-1):
            # reshape the tensor for SVD
            if site == 0:
                PSI = PSI.reshape(D, D**(L-1))
            else:
                PSI = PSI.reshape(r[site-1]*D, D**(L-site-1))
            # perform SVD
            U, S, Vh = np.linalg.svd(PSI, full_matrices=False)
            r.append(len(S)) # update bond dimension
            # normalize U for numerical stability
            U /= np.linalg.norm(U, axis=0)
            # append the leftmost tensor to MPS
            MPS_tensor.append(U)

            # update PSI for the next site
            PSI = np.dot(np.diag(S), Vh)

            # debugging print statements
            print("MPS for site {:}:".format(site+1))
            print("Shape of U:{:}".format(MPS_tensor[site].shape))
            print("shape of S:{:}".format(S.shape))
            print("SVD values:\n{:}".format(S))
            print("Shape of Vh:{:}".format(Vh.shape))
            print("shape of PSI:{:}".format(PSI.shape))

        # the right most MPS tensor should be PSI itself?
        MPS_tensor.append(PSI)

        # reshape MPS tensors
        for site in range(L):
            if site != 0 and site != L-1:
                MPS_tensor[site] = MPS_tensor[site].reshape(r[site-1], D, r[site])
            print("MPS site: {:}".format(site+1))
            print("Reshaped MPS tensors: {:}".format(MPS_tensor[site].shape))
        # store the decompose MPS tensor as an globally
        self.MPS_tensor_left = MPS_tensor

        return
