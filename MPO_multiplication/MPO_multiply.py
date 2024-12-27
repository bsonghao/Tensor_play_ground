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


class MPO_multiply(object):
    """
    The object implement the algorithms that efficientlu multply to MPOs
    """
    def __init__(self, D_MPO_1, D_MPO_2, d_MPO_1, d_MPO_2, L):
        """
        initialize a random MPO and MPS
        D_MPO_1: bond dimension of the first input MPO
        D_MPO_2: bond dimension of the second input MPO
        d_MPO_1: the incoming physical dimension of the first input MPO
        d_MPO_2: outcoming physical dimension of the second input MPO
        (the outcoming physical dimension of the first input MPO matches
         the incoming physical dimension of the second input MPO  )
        L: number of site
        """
        self.D_MPO_1 = D_MPO_1
        self.D_MPO_2 = D_MPO_2
        self.d_MPO_1 = d_MPO_1
        self.d_MPO_2 = d_MPO_2
        self.L = L

        # initial the input MPOs as python dictionary
        self.input_MPO_1 = {}
        self.input_MPO_2 = {}


        # construct the initial MPOs
        for site in range(L):
            left_bond_dim_1 = D_MPO_1 if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim_1 = D_MPO_1 if site < L - 1 else 1  # Right bond dimension is 1 for the last site

            left_bond_dim_2 = D_MPO_1 if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim_2 = D_MPO_1 if site < L - 1 else 1  # Right bond dimension is 1 for the last site

            shape_1 = [left_bond_dim_1, d_MPO_1, d_MPO_2, right_bond_dim_1]
            shape_2 = [left_bond_dim_2, d_MPO_2, d_MPO_2, right_bond_dim_2]

            self.input_MPO_1[site] = np.random.rand(*shape_1)
            self.input_MPO_2[site] = np.random.rand(*shape_2)

        # print input local operator for debug
        for site in range(L):
            print("site:{:}".format(site+1))
            print("first MPO tensor shape:{:}".format(self.input_MPO_1[site].shape))
            print("second MPO tensor shape:{:}".format(self.input_MPO_2[site].shape))

            # print("MPO tensor:\n{:}".format(self.input_MPO[site]))

    def _contract(self, input_MPO_1, input_MPO_2):
        """procedure the make contraction at each site to evaluate the product between a MPO and a MPS"""
        output_tensor = np.einsum('iabj,kbcl->ikacjl', input_MPO_1, input_MPO_2)

        return output_tensor

    def cal_MPO_MPO_product(self):
        """calcuate expectation value site by site"""
        output_MPO = {}
        for site in range(self.L):
            local_output_MPS = self._contract(self.input_MPO_1[site], self.input_MPO_2[site])

            # shrink the bond dimension of the new output mps
            left_bond_dim_MPO_1, right_bond_dim_MPO_1 = self.input_MPO_1[site].shape[0], self.input_MPO_1[site].shape[3]
            left_bond_dim_MPO_2, right_bond_dim_MPO_2 = self.input_MPO_2[site].shape[0], self.input_MPO_2[site].shape[3]
            new_shape = (left_bond_dim_MPO_1*left_bond_dim_MPO_2, self.d_MPO_1, self.d_MPO_2, right_bond_dim_MPO_1*right_bond_dim_MPO_2)
            local_output_MPS = local_output_MPS.reshape(*new_shape)

            # store the new MPS
            output_MPO[site] = local_output_MPS


        # print for debug purpose
        for site in output_MPO.keys():
            print("site:{:}".format(site+1))
            print("output MPS shape:{:}".format(output_MPO[site].shape))


        return output_MPO
