
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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPS_canonical/'))
sys.path.insert(0, project_dir)

# local import
from MPS_canonical import MPS_canonical

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    bond_dimension = 6
    num_state = 4
    num_site = 5

    tensor = MPS_canonical(bond_dimension, num_state, num_site)
    tensor.left_canonical()
    tensor.right_canonical()

    return

if (__name__ == '__main__'):
    main()
