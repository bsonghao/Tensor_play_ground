
import io
import time
import os
from os.path import abspath, join, dirname, basename
import sys
import cProfile
import pstats

# third party import
import numpy as np

# import the path to the package
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPO_muliplication/'))
sys.path.insert(0, project_dir)

# local import
from MPO_multiply import MPO_multiply

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    first_MPO_bond_dimension = 3
    second_MPO_bond_dimension = 2
    first_MPO_phys_dim = 4
    second_MPO_phys_dim = 6
    num_site = 5

    arges = (first_MPO_bond_dimension, second_MPO_bond_dimension, first_MPO_phys_dim, second_MPO_phys_dim, num_site)
    tensor = MPO_multiply(*arges)
    tensor.cal_MPO_MPO_product()

    return

if (__name__ == '__main__'):
    main()
