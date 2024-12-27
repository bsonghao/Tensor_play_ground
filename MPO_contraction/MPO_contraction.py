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

# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPS_canonical/'))
sys.path.insert(0, project_dir)

# local import
from  MPS_canonical import MPS_canonical

class MPO_contraction(MPS_canonical):
    """
    The object implement the algorithms that efficiently applying an MPO to ab MPS
    """
    def __init__(self, D_MPS, D_MPO, d_MPS, d_MPO, L):
        """
        initialize a random MPO and MPS
        D_MPS: bond dimension of the input MPS
        D_MPO: bond dimension of the input MPO
        d_MPS: physical dimension of the input MPS
        d_MPO: outcoming physical dimension of the input MPO
        (the incoming physical dimension of the input MPO matches the input MPS by definition)
        L: number of site
        """
        self.D_MPS = D_MPS
        self.D_MPO = D_MPO
        self.d_MPS = d_MPS
        self.d_MPO = d_MPO
        self.L = L

        # initial the input MPO as python dictionary
        self.input_MPO = {}

        # inherit instances from MPS_canonical class
        super(MPO_contraction, self).__init__(D_MPS, d_MPS, L)
        self.input_MPS = self.left_canonical()

        # construct the initial MPO
        for site in range(L):
            left_bond_dim = D_MPO if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim = D_MPO if site < L - 1 else 1  # Right bond dimension is 1 for the last site

            shape = [left_bond_dim, d_MPO, d_MPS, right_bond_dim]
            self.input_MPO[site] = np.random.rand(*shape)

        # print input local operator for debug
        for site in self.input_MPO.keys():
            print("site:{:}".format(site+1))
            print("MPO tensor shape:\n{:}".format(self.input_MPO[site].shape))
            # print("MPO tensor:\n{:}".format(self.input_MPO[site]))

    def _contract(self, input_MPO, input_MPS):
        """procedure the make contraction at each site to evaluate the product between a MPO and a MPS"""
        output_tensor = np.einsum('iabj,kbl->ikajl', input_MPO, input_MPS)

        return output_tensor

    def cal_MPO_MPO_product(self):
        """calcuate expectation value site by site"""
        output_MPS = {}
        for site in range(self.L):
            local_output_MPS = self._contract(self.input_MPO[site], self.input_MPS[site])

            # shrink the bond dimension of the new output mps
            left_bond_dim_MPO, right_bond_dim_MPO = self.input_MPO[site].shape[0], self.input_MPO[site].shape[3]
            left_bond_dim_MPS, right_bond_dim_MPS = self.input_MPS[site].shape[0], self.input_MPS[site].shape[2]
            new_shape = (left_bond_dim_MPO*left_bond_dim_MPS, self.d_MPO, right_bond_dim_MPO*right_bond_dim_MPS)
            local_output_MPS = local_output_MPS.reshape(*new_shape)

            # store the new MPS
            output_MPS[site] = local_output_MPS


        # print for debug purpose
        for site in output_MPS.keys():
            print("site:{:}".format(site+1))
            print("output MPS shape:{:}".format(output_MPS[site].shape))


        return output_MPS
