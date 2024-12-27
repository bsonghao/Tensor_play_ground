
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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPO_contractions/'))
sys.path.insert(0, project_dir)

# local import
from MPO_contraction import MPO_contraction

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    MPS_bond_dimension = 3
    MPO_bond_dimension = 2
    MPS_phys_dim = 4
    MPO_phys_dim = 6
    num_site = 5

    arges = (MPS_bond_dimension, MPO_bond_dimension, MPS_phys_dim, MPO_phys_dim, num_site)
    tensor = MPO_contraction(*arges)
    tensor.cal_MPO_MPO_product()

    return

if (__name__ == '__main__'):
    main()
