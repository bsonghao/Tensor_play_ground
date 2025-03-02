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

        if False:
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

    def check_result(self, input_MPS):
        """check if the decomposed MPS can restore the original tensor"""
        # form a python dictionary form alphabet
        alphabet = {}
        lower_start = ord('a')
        upper_start = ord('A')
        for i in range(26):
            alphabet[i] = chr(lower_start)
            lower_start += 1
        for i in range(26,52):
            alphabet[i] = chr(upper_start)
            upper_start += 1
        if False:
            # print for debug purpose
            for label in alphabet.keys():
                print("key for alphabet:{:}".format(label))
                print("corresponding letter:{:}".format(alphabet[label]))

        # initialize the input tensor with the same shape as the input tensor
        output_string = ""
        for site in range(self.L):
            left_bond_dim, phys_dim, right_bond_dim = input_MPS[site].shape

            if site != 0:
                # consecutively add node in the tensor train
                string = output_string
                string += "x,xyz->"
                string += output_string
                string += "yz"
                output_tensor = np.einsum(string, output_tensor, input_MPS[site]).copy()
            else:
                # elimate the dummy index
                output_tensor = input_MPS[site].squeeze().copy()

            # print(output_tensor.shape)
            output_string += alphabet[site]

        output_tensor = output_tensor.squeeze()

        # print("shape of the restored tensor:{:}".format(output_tensor.shape))
        # print("shape of the input tensor:{:}".format(self.input_tensor.shape))
        # print(abs(self.input_tensor-output_tensor))
        assert np.allclose(self.input_tensor, output_tensor)


        return


    def left_decompose(self):
        """algorithm to decompose the tensor to MPS from the left"""
        L, D = self.L, self.D

        # define an empty dictionary to store MPS tensor from left to the right
        MPS_tensor = {}

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
            # U /= np.linalg.norm(U, axis=0)
            # append the leftmost tensor to MPS
            MPS_tensor[site] = U

            # update PSI for the next site
            PSI = np.dot(np.diag(S), Vh)

            # debugging print statements
            print("MPS for site {:}:".format(site+1))
            print("Shape of U:{:}".format(MPS_tensor[site].shape))
            print("shape of S:{:}".format(S.shape))
            print("SVD values:\n{:}".format(S))
            print("Shape of Vh:{:}".format(Vh.shape))
            print("shape of PSI:{:}".format(PSI.shape))

        # the right most MPS tensor should be PSI itself
        MPS_tensor[L-1] = PSI

        # reshape MPS tensors
        for site in range(L):
            if site != 0 and site != L-1:
                MPS_tensor[site] = MPS_tensor[site].reshape(r[site-1], D, r[site]) # shape should be (left_bond_dim, phys_dim, right_bond_dim)
            # introduce dummy index at the edges
            elif site == 0:
                phys_dim, right_bond_dim = MPS_tensor[site].shape
                MPS_tensor[site] = MPS_tensor[site].reshape(1, phys_dim, right_bond_dim)
            else:
                left_bond_dim, phys_dim = MPS_tensor[site].shape
                MPS_tensor[site] = MPS_tensor[site].reshape(left_bond_dim, phys_dim, 1)

        # print for debug purpose
        if False:
            for site in MPS_tensor.keys():
                print("MPS site: {:}".format(site+1))
                print("Reshaped MPS tensors: {:}".format(MPS_tensor[site].shape))
                assert np.allclose(np.einsum('iaj,iak->jk', MPS_tensor[site], MPS_tensor[site]), np.eye(MPS_tensor[site].shape[2]))

        # store the decompose MPS tensor as an globally
        MPS_tensor_left = MPS_tensor

        # check the decomposed tensor
        self.check_result(MPS_tensor_left)

        return MPS_tensor_left

    def right_decompose(self):
        """algorithm to decompose the tensor to MPS from the left"""
        L, D = self.L, self.D

        # define an empty dictionary to store MPS tensor from left to the right
        MPS_tensor = {}

        # sweep from left to right to construct MPS tensors
        PSI = self.input_tensor.copy()
        # store bond dimension data
        r = {} # initial bond dimension
        for i in range(L-1):
            site = L - i -1
            # reshape the tensor for SVD
            if site == L-1:
                PSI = PSI.reshape(D**(L-1), D)
            else:
                print(PSI.shape)
                PSI = PSI.reshape(D**(site), r[site+1]*D)
            # perform SVD
            U, S, Vh = np.linalg.svd(PSI, full_matrices=False)
            r[site]=len(S) # update bond dimension
            # normalize U for numerical stability
            # U /= np.linalg.norm(U, axis=0)
            # append the leftmost tensor to MPS
            MPS_tensor[site] = Vh

            # update PSI for the next site
            PSI = np.dot(U, np.diag(S))

            # debugging print statements
            print("MPS for site {:}:".format(site+1))
            print("Shape of U:{:}".format(MPS_tensor[site].shape))
            print("shape of S:{:}".format(S.shape))
            print("SVD values:\n{:}".format(S))
            print("Shape of Vh:{:}".format(Vh.shape))
            print("shape of PSI:{:}".format(PSI.shape))

        # the left most MPS tensor should be PSI itself
        MPS_tensor[0] = PSI

        # reshape MPS tensors
        for site in range(L):
            if site != 0 and site != L-1:
                MPS_tensor[site] = MPS_tensor[site].reshape(r[site], D, r[site+1]) # shape should be (left_bond_dim, phys_dim, right_bond_dim)
            # introduce dummy index at the edges
            elif site == 0:
                phys_dim, right_bond_dim = MPS_tensor[site].shape
                MPS_tensor[site] = MPS_tensor[site].reshape(1, phys_dim, right_bond_dim)
            else:
                left_bond_dim, phys_dim = MPS_tensor[site].shape
                MPS_tensor[site] = MPS_tensor[site].reshape(left_bond_dim, phys_dim, 1)

        # print for debug purpose
        if False:
            for site in MPS_tensor.keys():
                print("MPS site: {:}".format(site+1))
                print("Reshaped MPS tensors: {:}".format(MPS_tensor[site].shape))
                assert np.allclose(np.einsum('aib,cib->ac', MPS_tensor[site], MPS_tensor[site]), np.eye(MPS_tensor[site].shape[0]))
        # store the decompose MPS tensor as an globally
        MPS_tensor_right = MPS_tensor

        # check the decomposed tensor
        self.check_result(MPS_tensor_right)

        return MPS_tensor_right
