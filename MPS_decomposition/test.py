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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/MPS_decomposition/'))
sys.path.insert(0, project_dir)

# local import
from MPS_decompose import MPS_decompose

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    num_site = 10
    num_state = 3

    tensor = MPS_decompose(num_site, num_state)
    tensor.left_decompose()
    tensor.right_decompose()

    return

if (__name__ == '__main__'):
    main()
