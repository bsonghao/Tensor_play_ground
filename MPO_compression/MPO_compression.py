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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPS_compression/'))
sys.path.insert(0, project_dir)


# local import
from MPS_compression import MPS_compression

class MPO_compression(MPS_compression):
    """This object implement the algorithms that approximate compression of an MPO"""
    def  __init__(self, D, d_1, d_2, L, D_truncate):
        """initialize an arbitary MPS and bring it to a left canonical form
        D_truncate: The truncation level of bond dimension in the compressed MPS
        d_1: incoming physical dimension
        d_2: outcoming physical dimension
        D: original bond dimension
        """
        self.d_1 = d_1
        self.d_2 = d_2

        # initialized the input MPO (with flatted physical dimensions) and left orthnormalized
        super(MPO_compression, self).__init__(D, d_1*d_2, L, D_truncate)

        # restore the physcial dimensions of the original MPO
        self.input_MPO = {}
        for site in range(L):
            left_bond_dim, phys_dim, right_bond_dim = self.left_canonical_MPS[site].shape
            self.input_MPO[site] = self.left_canonical_MPS[site].reshape(left_bond_dim, d_1, d_2, right_bond_dim)

        # print for debug purpose
        print("Input MPO:")
        for site in self.input_MPO.keys():
            print("Site: {:}".format(site+1))
            print("MPO shape: {:}".format(self.input_MPO[site].shape))

    def MPO_iterative_compression(self):
        """iterative compress the MPO (calling MPS compression subroutines)"""

        # compress the input MPO
        output_MPO = self.iterative_compress()
        # restore the fattened physical dimension
        d_1, d_2 = self.d_1, self.d_2
        for site in output_MPO.keys():
            left_bond_dim, phys_dim, right_bond_dim = output_MPO[site].shape
            output_MPO[site] = output_MPO[site].reshape(left_bond_dim, d_1, d_2, right_bond_dim)

        # print output MPO for debug purpose
        print("Compressed MPO:")
        for site in output_MPO.keys():
            print("site {:}:".format(site+1))
            print("MPO shape:{:}".format(output_MPO[site].shape))

        return output_MPO
