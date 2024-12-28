
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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPO_compression/'))
sys.path.insert(0, project_dir)

# local import
from MPO_compression import MPO_compression

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    bond_dimension = 6
    truncated_bond_dimension = 2
    incoming_phys_dim = 3
    outcoming_phys_dim = 4
    num_site = 5

    arges = (bond_dimension, incoming_phys_dim, outcoming_phys_dim, num_site, truncated_bond_dimension)
    tensor = MPO_compression(*arges)
    tensor.MPO_iterative_compression()

    return

if (__name__ == '__main__'):
    main()