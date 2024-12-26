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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPS_decomposition/'))
sys.path.insert(0, project_dir)

# local import
from MPS_decompose import MPS_decompose

class MPO_decomposite(MPS_decompose):
    """This class implement methods that decompose an arbitrary high dimensional tensor into a MPO"""
    def __init__(self, D_1, D_2, L):
        """initialize the input tensor
        D_1: the outcoming physical dimension
        D_2: the incomping physical dimension
        L: number of site
        """
        self.D_1 = D_1
        self.D_2 = D_2

        # initialize the initial tensor (with physical dimensions flattened)
        phys_dim = D_1*D_2
        # X = MPS_decompose(L, phys_dim)
        super(MPO_decomposite, self).__init__(L, phys_dim)

    def MPO_left_decompose(self):
        """decompose the tensor into MPO from the left"""

        print("*** left MPO decomposition procedure start")

        decomposed_MPO = self.left_decompose()
        # restore the physical dimension of the original tensor
        for site in decomposed_MPO.keys():
            left_bond_dim, phys_dim, right_bond_dim = decomposed_MPO[site].shape
            decomposed_MPO[site] = decomposed_MPO[site].reshape(left_bond_dim, self.D_1, self.D_2, right_bond_dim)

        #print the decomposed MPO for debug purpose
        for site in decomposed_MPO.keys():
            print("site {:}:".format(site))
            print("shape of MPO:{:}".format(decomposed_MPO[site].shape))

        print("*** left MPO decomposition procedure terminate")

        return decomposed_MPO

    def MPO_right_decompose(self):
        """decompose the tensor into MPO from the right"""

        print("*** right MPO decomposition procedure start")

        decomposed_MPO = self.right_decompose()

        # restore the physical dimension of the original tensor
        for site in decomposed_MPO.keys():
            left_bond_dim, phys_dim, right_bond_dim = decomposed_MPO[site].shape
            decomposed_MPO[site] = decomposed_MPO[site].reshape(left_bond_dim, self.D_1, self.D_2, right_bond_dim)

        #print the decomposed MPO for debug purpose
        for site in decomposed_MPO.keys():
            print("site {:}:".format(site))
            print("shape of MPO:{:}".format(decomposed_MPO[site].shape))

        print("*** right MPO decomposition procedure terminate")

        return decomposed_MPO
