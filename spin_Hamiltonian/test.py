
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
    h = 1.
    Jz = 1.
    chemical_shift = np.random.rand(L)

    # step 1:  using large h to "prepare" as initial state with large anisotropy term Hamiltonian
    tensor = spin_Hamiltonian(num_site=L, J=J, Jz=Jz, h=10, chemical_shift=chemical_shift)
    initial_MPS = tensor.ground_state_search(num_sweep=2, D=4)

    # step 2: using the prepared initial state with large anisotropy as to state the time evolution
    tensor = spin_Hamiltonian(num_site=L, J=J, Jz=Jz, h=h, chemical_shift=chemical_shift)
    tensor.TDVP_evolution(t_final=10, num_sweep=200, D=4, imagine_t=False)


    return

if (__name__ == '__main__'):
    main()
