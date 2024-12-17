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
from MPS_canonical import MPS_canonical

class MPS_compression(MPS_canonical):
    """This object implement the algorithms that approximate compression of an MPS"""
    def  __init__(self, D, d, L, D_truncate):
        """initialize an arbitary MPS and bring it to a left canonical form
        D_truncate: The truncation level of bond dimension in the compressed MPS
        """
        self.D_truncate = D_truncate
        super(MPS_compression, self).__init__(D, d, L)

        # call the subroutine to bring te initial MPS into left canonical form
        self.left_canonical()

    def SVD_compress(self):
        """
        Implement method that compresses the MPS by SVD
        """
        D, d, L, D_truncate = self.D, self.d, self.L, self.D_truncate
        def _local_compression(input_tensor, i):
            """perform SVD on each local site and contructed the compressed tensor"""
            left_bond_dim, phys_dim, right_bond_dim = input_tensor.shape
            # reshape the input tensor
            input_tensor = input_tensor.reshape(left_bond_dim, phys_dim*right_bond_dim)
            # SVD the reshaped input tensor
            U, S, B = np.linalg.svd(input_tensor, full_matrices=False)


            # determine truncated dimension
            truncation = min(D_truncate, len(S))
            S_truncate = S[:truncation]
            B_truncate = B[:truncation,:]
            U_truncate = U[:,:truncation]

            # rescale the single value to satisfy the normalization condition
            S_truncate = S_truncate/sum(S_truncate)


            # reshape the compressed tensor
            output_tensor = B_truncate.reshape(truncation, phys_dim, right_bond_dim)

            # print for debug propurse
            print("original single value at site {:}:\n{:}".format(i+1, S))
            print("compressed single value at site {:}:\n{:}".format(i+1, S_truncate))

            return output_tensor, U_truncate, S_truncate

        # loop over each site to perform the MPS compression
        self.SVD_compressed_MPS = {}
        for i in range(L):
            site = L-i-1
            # calcuate input tensor to perform SVD at each site
            if site == L-1:
                M = self.left_canonical_MPS[site]
            else:
                M = np.einsum('iaj,jk,k->iak',self.left_canonical_MPS[site], U_tilde, S_tilde)

            # perform compression at each site
            B_tilde, U_tilde, S_tilde = _local_compression(M, site)

            # store the compressed MPS at each site
            self.SVD_compressed_MPS[site] = B_tilde

        # print and check the compressed tensor for debuging purpose
        for site in range(L):
            tensor = self.SVD_compressed_MPS[site]
            left_bond_dim, phys_dim, right_bond_dim = tensor.shape
            print("Site {:}:".format(site+1))
            print("shape:{:}".format(tensor.shape))
            # check if the tensor is left-canonical
            assert np.allclose(np.einsum('aib,cib->ac', tensor, tensor), np.eye(left_bond_dim))
            # print("tensor:\n{:}".format(self.right_canonical_MPS[site]))


        return
