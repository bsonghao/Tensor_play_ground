# system imports
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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPO_decomposition/'))
sys.path.insert(0, project_dir)

# local import
from MPO_decomposite import MPO_decomposite

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    num_site = 5
    num_incoming_state = 3
    num_outcomping_state = 4

    tensor = MPO_decomposite(num_incoming_state, num_outcomping_state, num_site)
    tensor.MPO_left_decompose()
    tensor.MPO_right_decompose()


    return

if (__name__ == '__main__'):
    main()
