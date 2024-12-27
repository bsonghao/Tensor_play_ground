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

class MPS_canonical(object):
    """
    The object implement the algorithm that bring an arbitary MPS into caonical form
    """
    def __init__(self, D, d, L):
        """
        initialize a random MPO and MPS
        D: bond dimension of the MPS
        d: physical dimension at each site
        L: number of site
        """
        self.D = D
        self.d = d
        self.L = L

        # initial the input MPS and the local operator as python dictionary
        self.input_MPS = {}

        # construct an arbitary MPS
        for site in range(L):
            # Determine dimensions of the current tensor
            left_bond_dim = D if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim = D if site < L - 1 else 1  # Right bond dimension is 1 for the last site

            # Generate a random tensor of shape (left_bond_dim, phys_dim, right_bond_dim)
            tensor = np.random.randn(left_bond_dim, d, right_bond_dim)

            self.input_MPS[site] = tensor

        # print input MPS for debug
        for site in self.input_MPS.keys():
            print("site:{:}".format(site+1))
            print("MPS matrix shape:{:}".format(self.input_MPS[site].shape))
            # print("MPS matrix:\n{:}".format(self.input_MPS[site]))

    def left_canonical(self, input_MPS=0):
        """procedure to bring the MPS into left canonical form"""
        def _local_canonical(input_tensor):
            """produce left-canonical matrix at each site"""
            left_bond_dim, phys_dim, right_bond_dim = input_tensor.shape
            # reshape the input tensor into the shape (left_bond_dim * phys_dim, right_bond_dim)
            input_tensor = input_tensor.reshape(left_bond_dim*phys_dim, right_bond_dim)
            # SVD the reshaped tensor
            A, S, Vh = np.linalg.svd(input_tensor, full_matrices=False)
            # reshape the decomposed tensor in to the original shape
            # change right bond dimension for base cases
            if left_bond_dim == 1:
                right_bond_dim = min(phys_dim, right_bond_dim)
            else:
                pass
            output_tensor = A.reshape(left_bond_dim, phys_dim, right_bond_dim)

            return output_tensor, S, Vh

        if False:
            input_MPS = self.input_MPS
        else:
            pass

        D, d, L = self.D, self.d, self.L
        left_canonical_MPS = {}
        # loop over each site of the MPS to form left-canonical MPS
        for site in range(L):
            # base case
            if site == 0:
                input_tensor = input_MPS[site]
            else:
                input_tensor = np.einsum('s,sa,aib->sib', S, V, input_MPS[site])

            A, S, V = _local_canonical(input_tensor)

            # site the decomponsed tensor at each site
            left_canonical_MPS[site] = A

        # check if the procedure produce the lelf-canonical MPS
        print("Left canonical MPS:")
        for site in range(L):
            tensor =left_canonical_MPS[site]
            left_bond_dim, phys_dim, right_bond_dim = tensor.shape
            assert np.allclose(np.einsum('bia,bic->ac', tensor, tensor), np.eye(right_bond_dim))
            print("Site {:}:".format(site+1))
            print("shape:{:}".format(left_canonical_MPS[site].shape))
            # print("tensor:\n{:}".format(self.left_canonical_MPS[site]))

        return left_canonical_MPS

    def right_canonical(self, input_MPS=0):
        """procedure to bring the MPS into right-canonical form"""
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

        if True:
            input_MPS = self.input_MPS
        else:
            pass

        D, d, L = self.D, self.d, self.L
        self.right_canonical_MPS = {}
        # loop over each site of the MPS to form left-canonical MPS
        for i in range(L):
            site = L-i-1
            # base case
            if site == L-1:
                input_tensor = input_MPS[site]
            else:
                input_tensor = np.einsum('aib,bs,s->ais', input_MPS[site], U, S)

            B, S, U = _local_canonical(input_tensor)

            # site the decomponsed tensor at each site
            self.right_canonical_MPS[site] = B

        # check if the procedure produce the lelf-canonical MPS
        print("Right canonical MPS:")
        for site in range(L):
            tensor = self.right_canonical_MPS[site]
            left_bond_dim, phys_dim, right_bond_dim = tensor.shape
            assert np.allclose(np.einsum('aib,cib->ac', tensor, tensor), np.eye(left_bond_dim))
            print("Site {:}:".format(site+1))
            print("shape:{:}".format(self.right_canonical_MPS[site].shape))
            print("tensor:\n{:}".format(self.right_canonical_MPS[site]))

        return self.right_canonical_MPS
