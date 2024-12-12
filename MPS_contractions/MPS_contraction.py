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

class MPS_contraction(object):
    """
    The object implement the contraction algorithms that efficiently calcuate overlaps, expectation value and matrix elements for MPS
    """
    def __init__(self, D, d, L):
        """
        initialize a random MPO and MPS
        D: bond dimension of the MPS
        (Assume that the MPO and MPS have the same physical dimension and number of sites)
        d: physical dimension at each site
        L: number of site
        """
        self.D = D
        self.d = d
        self.L = L

        # initial the input MPS and the local operator as python dictionary
        self.input_MPS = {}
        self.input_OP = {}

        # construct the initial MPS
        for site in range(L):
            # Determine dimensions of the current tensor
            left_bond_dim = D if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim = D if site < L - 1 else 1  # Right bond dimension is 1 for the last site

            # Generate a random tensor of shape (left_bond_dim * phys_dim, right_bond_dim)
            tensor = np.random.randn(left_bond_dim * d, right_bond_dim)

            # Perform QR decomposition to orthonormalize
            q, _ = np.linalg.qr(tensor)

            assert np.allclose(np.dot(q.T, q), np.eye(right_bond_dim))

            # Reshape Q into (left_bond_dim, phys_dim, right_bond_dim)
            tensor = q.reshape(left_bond_dim, d, right_bond_dim)

            self.input_MPS[site] = tensor

        # print input MPS for debug
        for site in self.input_MPS.keys():
            print("site:{:}".format(site+1))
            print("MPS matrix shape:{:}".format(self.input_MPS[site].shape))
            print("MPS matrix:\n{:}".format(self.input_MPS[site]))

        # construct the initial local operator O[1]O[2]...O[L]
        norm_flag = True #all O[i] = I if we want to evaluate the norm
        for site in range(L):
            if norm_flag:
                self.input_OP[site] = np.eye(d)
            else:
                O = np.random.rand(d,d)
                self.input_OP[site] = 0.5 * (O.T + O) # assume that the local operator on each site is a Hermitian
        # print input local operator for debug
        for site in self.input_OP.keys():
            print("site:{:}".format(site+1))
            print("local operator matrix:\n{:}".format(self.input_OP[site]))

    def _contract(self, input_tensor, site):
        """procedure the make contraction at each site to evaluate the expectation value of a local operator"""
        output_tensor = np.einsum('ik,iaj,ab,kbl->jl',input_tensor, self.input_MPS[site], self.input_OP[site], self.input_MPS[site])

        return output_tensor

    def cal_expetation(self):
        """calcuate expectation value site by site"""
        D, d, L = self.D, self.d, self.L
        expectation_value = 0
        for site in range(L):
            if site == 0:
                # handle base case
                expectation_value = np.einsum('jai,ab,jbk->ik', self.input_MPS[site], self.input_OP[site], self.input_MPS[site]).copy()
            else:
                expectation_value = self._contract(expectation_value, site).copy()
        # reduce the dummy index
        print(expectation_value.shape)
        expectation_value = np.squeeze(expectation_value)
        print("expectation value:\n{:}".format(expectation_value))

        return expectation_value
