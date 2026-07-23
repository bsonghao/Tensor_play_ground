
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
project_dir = abspath(join(dirname(__file__), '/Users/pauliebao/Tensor_play_ground/spin_Hamiltonian/'))
sys.path.insert(0, project_dir)

# local import
from spin_Hamiltonian import spin_Hamiltonian

def main():
    """main function for MPS decomposition algorithm"""
    # input tensor shape
    L = 10
    J = 1.
    h = 0.
    Jz = 0.


    tensor = spin_Hamiltonian(L, J, Jz, h)
    tensor.ground_state_search(num_sweep=2, D=14)


    return

if (__name__ == '__main__'):
    main()
