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
        self.left_canonical_MPS = self.left_canonical(self.input_MPS)

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
            # print("original single value at site {:}:\n{:}".format(i+1, S))
            # print("compressed single value at site {:}:\n{:}".format(i+1, S_truncate))

            return output_tensor, U_truncate, S_truncate

        # loop over each site to perform the MPS compression
        SVD_compressed_MPS = {}
        for i in range(L):
            site = L-i-1
            # calcuate input tensor to perform SVD at each site
            if site == L-1:
                M = self.left_canonical_MPS[site].copy()
            else:
                M = np.einsum('iaj,jk,k->iak',self.left_canonical_MPS[site], U_tilde, S_tilde).copy()

            # perform compression at each site
            B_tilde, U_tilde, S_tilde = _local_compression(M, site)

            # store the compressed MPS at each site
            SVD_compressed_MPS[site] = B_tilde

        # print and check the compressed tensor for debuging purpose
        print("SVD compressed tensor:")
        for site in range(L):
            tensor = SVD_compressed_MPS[site]
            left_bond_dim, phys_dim, right_bond_dim = tensor.shape
            print("Site {:}:".format(site+1))
            print("shape:{:}".format(tensor.shape))
            # check if the tensor is left-canonical
            assert np.allclose(np.einsum('aib,cib->ac', tensor, tensor), np.eye(left_bond_dim))
            # print("tensor:\n{:}".format(self.right_canonical_MPS[site]))


        return SVD_compressed_MPS

    def iterative_compress(self):
        """implement method that compress a MPS from variational optimize the matrix parameterize on each site"""
        def _local_compression(decomposed_MPS, site):
            """variationally compress the MPS at each local site"""
            def _contract(input_tensor, input_MPS, site_i, right=False):
                """calculate overlaps at each local site"""
                if right:
                    output_tensor = np.einsum('ik,jai,lak->jl',input_tensor, self.left_canonical_MPS[site_i], input_MPS[site_i])
                else:
                    output_tensor = np.einsum('ik,iaj,kal->jl',input_tensor, self.left_canonical_MPS[site_i], input_MPS[site_i])
                return output_tensor

            def _cal_left_tensor():
                """calculate left tensor L"""
                # loop over all MPS tensors to the left
                for i in range(site):
                    if i == 0:
                        left_tensor = np.einsum('iaj,iak->jk', self.left_canonical_MPS[i], decomposed_MPS[i]).copy()
                    else:
                        left_tensor = _contract(left_tensor, decomposed_MPS, i).copy()

                return left_tensor

            def _cal_right_tensor():
                """calculate right tensor R"""
                # loop over all MPS tensors to the right
                for i in range(site+1, L):
                    site_i = L + site - i # start from the right most tensor
                    if site_i == L-1:
                        right_tensor = np.einsum('iaj,kaj->ik', self.left_canonical_MPS[site_i], decomposed_MPS[site_i]).copy()
                        print(right_tensor.shape)
                    else:
                        right_tensor = _contract(right_tensor, decomposed_MPS, site_i, right=True).copy()

                return right_tensor

            # handle edge cases
            if site == 0:
                R_tensor = _cal_right_tensor()
                output_tensor = np.einsum('ji,kaj->kai', R_tensor, self.left_canonical_MPS[site])
            elif site == L-1:
                L_tensor = _cal_left_tensor()
                output_tensor = np.einsum('ji,jak->iak', L_tensor, self.left_canonical_MPS[site])
            else:
                # calculate left tensor
                L_tensor = _cal_left_tensor()
                # calculate right tensor
                R_tensor = _cal_right_tensor()

                # calculate variational optimized local MPS tensor
                output_tensor = np.einsum('ji,lk,jal->iak', L_tensor, R_tensor, self.left_canonical_MPS[site])

            return output_tensor

        def _local_canonical(input_tensor):
            """produce right-canonical matrix at each site"""
            left_bond_dim, phys_dim, right_bond_dim = input_tensor.shape
            # reshape the input tensor into the shape (left_bond_dim * phys_dim, right_bond_dim)
            input_tensor = input_tensor.reshape(left_bond_dim, right_bond_dim*phys_dim)
            # SVD the reshaped tensor
            U, S, B = np.linalg.svd(input_tensor, full_matrices=False)
            # reshape the decomposed tensor in to the original shape
            # change left bond dimension for base cases
            if right_bond_dim == 1:
                left_bond_dim = min(phys_dim, left_bond_dim)
            else:
                pass
            output_tensor = B.reshape(left_bond_dim, phys_dim, right_bond_dim)

            return output_tensor, S, U

        print("***Start tensor compression procedure")

        D, d, L, D_truncate = self.D, self.d, self.L, self.D_truncate
        # using SVD compressed tensor as an initial guess
        initial_MPS = self.SVD_compress()
        # bring the SVD compressed tesnor into a left canonical form
        initial_MPS = self.left_canonical(initial_MPS)

        local_decomposed_MPS = {}
        iterative_decomposed_MPS = {}

        # loop over each site for local compression
        for i in range(L):
            local_site = L-i-1
            # firstly bring the left canonical MPS into mixed canonical form at each local site
            # base case
            if local_site == L-1:
                input_tensor = initial_MPS[local_site]
            else:
                input_tensor = np.einsum('aib,bs,s->ais', initial_MPS[local_site], U, S)

            B, S, U = _local_canonical(input_tensor)
            local_decomposed_MPS[local_site] = B
            # handle edge case
            if local_site == 0 or local_site == L-1:
                if local_site == L-1:
                    for j in range(local_site-1):
                        local_decomposed_MPS[j] = initial_MPS[j]
                    local_decomposed_MPS[local_site-1] = input_tensor
                iterative_decomposed_MPS[local_site] = _local_compression(local_decomposed_MPS, local_site)

            else:
                local_decomposed_MPS[local_site] = B
                local_decomposed_MPS[local_site-1] = input_tensor
                for j in range(local_site-1):
                    local_decomposed_MPS[j] = initial_MPS[j]

                # variationally compress MPS at each local site
                iterative_decomposed_MPS[local_site] = _local_compression(local_decomposed_MPS, local_site-1)

        if True:
            # print for debug purpose
            print("Initial guess of the MPS (obtained from svd)")
            for site in initial_MPS.keys():
                print("site: {:}".format(site+1))
                print("shape of the tensor:{:}".format(initial_MPS[site].shape))

            print("Original MPS")
            for site in self.left_canonical_MPS.keys():
                print("site: {:}".format(site+1))
                print("shape of the tensor:{:}".format(self.left_canonical_MPS[site].shape))

            print("Compressed tensor:")
            for site in iterative_decomposed_MPS.keys():
                print("site: {:}".format(site+1))
                print("shape of the tensor:{:}".format(iterative_decomposed_MPS[site].shape))

        print("***Tensor compression procedure terminate")

        return iterative_decomposed_MPS
