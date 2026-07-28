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


class spin_Hamiltonian(object):
    """This class initialize a spin Hamiltonian in the form of MPO and perform munipulation on it"""
    def __init__(self, num_site, J, Jz, h, chemical_shift=None):
        """
        initiize model parameter of the Hamiltonian and bring it into the MPO form

        chemical_shift: optional array-like of length num_site giving a
        per-site weight for the total magnetization (ZULF signal) operator
        sum_i chemical_shift[i] * S_z(i); defaults to uniform weight 1
        (plain total magnetization) if not provided. This only affects the
        magnetization operator used for observables, not the propagating
        Hamiltonian self.H.
        """
        self.L = num_site
        self.J = J
        self.Jz = Jz
        self.h = h
        if chemical_shift is None:
            chemical_shift = np.ones(num_site)
        self.chemical_shift = np.asarray(chemical_shift, dtype=float)
        if self.chemical_shift.shape != (num_site,):
            raise ValueError(f"chemical_shift must have shape ({num_site},), got {self.chemical_shift.shape}")

        # define Pauli matrices
        self.S_x = np.array([[0, 1], [1, 0]], dtype=complex)* 0.5 # set hbar =1
        self.S_y = np.array([[0, -1j], [1j, 0]], dtype=complex) * 0.5 # set hbar =1
        self.S_z = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5 # set hbar =1

        self.S_plus = (self.S_x.copy() + 1j*self.S_y.copy())
        self.S_minus = (self.S_x.copy() - 1j*self.S_y.copy())

        print("Pauli matrices:")
        print(f'sigma x:\n{self.S_x}')
        print(f'sigma y:\n{self.S_y}')
        print(f'sigma z:\n{self.S_z}')
        print(f'sigma plus:\n{self.S_plus}')
        print(f'sigma minus:\n{self.S_minus}')

        # initialize the spin Hamiltonian
        self.H = {}
        for site in range(self.L):
            # encode the spin Hamilotnian into MPO form (Eq(182) Schollwock chapter 6 )
            left_bond_dim = 5 if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim = 5 if site < self.L - 1 else 1  # Right bond dimension is 1 for the last site

            phys_dim = 2

            self.H[site] = np.zeros((left_bond_dim, phys_dim, phys_dim, right_bond_dim), dtype=complex)

            if site !=0 and site != self.L-1:
                self.H[site][0,:,:,0] += np.eye(phys_dim, dtype=complex)
                self.H[site][1,:,:,0] += self.S_plus
                self.H[site][2,:,:,0] += self.S_minus
                self.H[site][3,:,:,0] += self.S_z
                self.H[site][4,:,:,0] -= self.h * self.S_z
                self.H[site][4,:,:,1] -= self.J/2 * self.S_minus
                self.H[site][4,:,:,2] -= self.J/2 * self.S_plus
                self.H[site][4,:,:,3] += self.Jz * self.S_z
                self.H[site][4,:,:,4] += np.eye(phys_dim, dtype=complex)

            elif site == self.L-1:
                self.H[site][0,:,:,0]+= np.eye(phys_dim, dtype=complex)
                self.H[site][1,:,:,0] += self.S_plus
                self.H[site][2,:,:,0] += self.S_minus
                self.H[site][3,:,:,0] += self.S_z
                self.H[site][4,:,:,0] -= self.h * self.S_z

            else:
                self.H[site][0,:,:,0] -= self.h * self.S_z
                self.H[site][0,:,:,1] -= self.J/2 * self.S_minus
                self.H[site][0,:,:,2] -= self.J/2 * self.S_plus
                self.H[site][0,:,:,3] += self.Jz * self.S_z
                self.H[site][0,:,:,4] += np.eye(phys_dim, dtype=complex)


            print(f'spin Hamiltonian site {site+1}:\n{self.H[site].shape}')

        # build the MPO representation of the total magnetization operator sum_i S_z(i)
        self.Sz_total_MPO = self._build_total_Sz_MPO()

    def _build_total_Sz_MPO(self):
        """construct the MPO for the (chemical-shift-weighted) total
        magnetization / ZULF-signal operator sum_i chemical_shift[i] * S_z(i)

        uses the same bond-state convention as self.H: bond index 0 means
        "a (weighted) S_z has already been applied, just propagate identity
        from here on", bond index 1 means "S_z has not been applied yet"
        """
        L, phys_dim = self.L, 2
        Sz_MPO = {}
        for site in range(L):
            left_bond_dim = 2 if site > 0 else 1
            right_bond_dim = 2 if site < L - 1 else 1

            weighted_Sz = self.chemical_shift[site] * self.S_z

            Sz_MPO[site] = np.zeros((left_bond_dim, phys_dim, phys_dim, right_bond_dim), dtype=complex)

            if site != 0 and site != L - 1:
                Sz_MPO[site][0, :, :, 0] += np.eye(phys_dim, dtype=complex)
                Sz_MPO[site][1, :, :, 0] += weighted_Sz
                Sz_MPO[site][1, :, :, 1] += np.eye(phys_dim, dtype=complex)
            elif site == L - 1:
                Sz_MPO[site][0, :, :, 0] += np.eye(phys_dim, dtype=complex)
                Sz_MPO[site][1, :, :, 0] += weighted_Sz
            else:
                Sz_MPO[site][0, :, :, 0] += weighted_Sz
                Sz_MPO[site][0, :, :, 1] += np.eye(phys_dim, dtype=complex)

        return Sz_MPO

    def _initialize_mps_all_up(self, D):
        """initialize a product-state MPS with every spin aligned along +Z

        the physical basis is ordered so that index 0 corresponds to the
        +1/2 eigenstate of S_z (see self.S_z), so each site tensor is a
        pure |up> state embedded (zero-padded) into bond dimension D
        """
        L = self.L
        initial_MPS = {}
        for site in range(L):
            left_bond_dim = D if site > 0 else 1
            right_bond_dim = D if site < L - 1 else 1

            tensor = np.zeros((left_bond_dim, 2, right_bond_dim), dtype=complex)
            tensor[0, 0, 0] = 1.0

            initial_MPS[site] = tensor

        for site in initial_MPS.keys():
            print("site:{:}".format(site+1))
            print("Initial MPS matrix shape:{:}".format(initial_MPS[site].shape))

        return initial_MPS

    def _initialize_mps(self, D):
        """initialize a random MPS"""
        L = self.L
        # initial the input MPS and the local operator as python dictionary
        initial_MPS = {}
        # construct an arbitary MPS
        for site in range(L):
            # Determine dimensions of the current tensor
            left_bond_dim = D if site > 0 else 1  # Left bond dimension is 1 for the first site
            right_bond_dim = D if site < L - 1 else 1  # Right bond dimension is 1 for the last site

            # Generate a random tensor of shape (left_bond_dim, phys_dim, right_bond_dim)
            tensor = np.random.randn(left_bond_dim, 2, right_bond_dim)

            initial_MPS[site] = tensor

        # print input MPS for debug

        for site in initial_MPS.keys():
            print("site:{:}".format(site+1))
            print("Initial MPS matrix shape:{:}".format(initial_MPS[site].shape))
            # print("MPS matrix:\n{:}".format(self.input_MPS[site]))

        return initial_MPS

    def _right_canonical(self, input_MPS, D):
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
            left_bond_dim = min(right_bond_dim*phys_dim, left_bond_dim)

            output_tensor = B.reshape(left_bond_dim, phys_dim, right_bond_dim)

            return output_tensor, S, U


        d, L = 2, self.L
        right_canonical_MPS = {}
        # loop over each site of the MPS to form right-canonical MPS
        for i in range(L):
            site_i = L-i-1
            # base case
            if site_i == L-1:
                input_tensor = input_MPS[site_i]
            else:
                input_tensor = np.einsum('aib,bs,s->ais', input_MPS[site_i], U, S)

            # if site_i != 0:
            B, S, U = _local_canonical(input_tensor)
            right_canonical_MPS[site_i] = B.copy()

            # else:
                # right_canonical_MPS[site_i] = input_tensor.copy()

        # check if the procedure produce the lelf-canonical MPS
        if True:
            # print("Right canonical MPS:")
            for site in range(L):
                tensor = right_canonical_MPS[site].copy()
                left_bond_dim, phys_dim, right_bond_dim = tensor.shape
                # if site != 0:
                assert np.allclose(np.einsum('aib,cib->ac', tensor, tensor), np.eye(left_bond_dim))
                print("Right canonical MPS Site {:}:".format(site+1))
                print("shape:{:}".format(right_canonical_MPS[site].shape))
                # print("tensor:\n{:}".format(right_canonical_MPS[site]))
        return right_canonical_MPS

    def _left_canonical(self, input_MPS, D):
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
            right_bond_dim = min(left_bond_dim*phys_dim, right_bond_dim)

            output_tensor = A.reshape(left_bond_dim, phys_dim, right_bond_dim)

            return output_tensor, S, Vh

        d, L = 2, self.L
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
        if True:
            print("Left canonical MPS:")
            for site in range(L):
                tensor =left_canonical_MPS[site]
                left_bond_dim, phys_dim, right_bond_dim = tensor.shape
                assert np.allclose(np.einsum('bia,bic->ac', tensor, tensor), np.eye(right_bond_dim))
                print("Site {:}:".format(site+1))
                print("shape:{:}".format(left_canonical_MPS[site].shape))
                # print("tensor:\n{:}".format(self.left_canonical_MPS[site]))

        return left_canonical_MPS


    def _cal_expectation(self, input_MPS, input_MPO):
        """calcuate expectation value site by site"""
        L = self.L
        def _contract(input_tensor, site):
            """procedure the make contraction at each site to evaluate the expectation value of a local operator"""
            output_tensor = np.einsum('ijk,ial,jabm,kbn->lmn',input_tensor, input_MPS[site].conj(), input_MPO[site], input_MPS[site])
            return output_tensor

        expectation_value = 0
        for site in range(L):
            if site == 0:
                # handle base case
                expectation_value = np.einsum('ial,jabm,kbn->ijklmn', input_MPS[site].conj(), input_MPO[site], input_MPS[site]).squeeze()
            else:
                expectation_value = _contract(expectation_value, site).copy()
        # reduce the dummy index
        expectation_value = np.squeeze(expectation_value)
        if False:  # print for debug
            print("expectation value:\n{:}".format(expectation_value))
        return expectation_value


    def _cal_MPO_MPO_product(self, MPO_1, MPO_2):
        """calcuate the product between the two input MPOs site by site"""
        d, L = 2, self.L
        def _contract(input_MPO_1, input_MPO_2):
           """procedure the make contraction at each site to evaluate the product two a MPOs"""
           output_tensor = np.einsum('iabj,kbcl->ikacjl', input_MPO_1, input_MPO_2).copy()
           return output_tensor

        output_MPO = {}
        for site in range(L):
            local_output_MPS = _contract(MPO_1[site], MPO_2[site])

            # shrink the bond dimension of the new output mps
            left_bond_dim_MPO_1, right_bond_dim_MPO_1 = MPO_1[site].shape[0], MPO_1[site].shape[3]
            left_bond_dim_MPO_2, right_bond_dim_MPO_2 = MPO_2[site].shape[0], MPO_2[site].shape[3]
            new_shape = (left_bond_dim_MPO_1*left_bond_dim_MPO_2, d, d, right_bond_dim_MPO_1*right_bond_dim_MPO_2)
            local_output_MPS = local_output_MPS.reshape(*new_shape)

            # store the new MPS
            output_MPO[site] = local_output_MPS


        # print for debug purpose
        if False: # print for debug
            for site in output_MPO.keys():
                print("site:{:}".format(site+1))
                print("output MPS shape:{:}".format(output_MPO[site].shape))

        return output_MPO

    def _cal_variance(self, input_MPO, input_MPS):
        """calcuate the variance (<O^2> - <O>^2) of the input MPO w.r.t the input MPS """
        # calcuate H^2
        H_square = self._cal_MPO_MPO_product(self.H, self.H)

        # calcuate <H^2>
        H_square_avg = self._cal_expectation(input_MPS, H_square)

        # calcuate <H>
        H_avg = self._cal_expectation(input_MPS, self.H)

        # calculate variance
        H_var = H_square_avg - H_avg**2

        return H_var

    def _contract(self, input_MPS, input_tensor, site_y, right=False):
        """procedure the make contraction at each site to evaluate the expectation value of a local operator"""
        if right:
            output_tensor = np.einsum('jnl,iaj,mabn,kbl->imk',input_tensor, input_MPS[site_y].conj(), self.H[site_y], input_MPS[site_y])

        else:
            output_tensor = np.einsum('imk,iaj,mabn,kbl->jnl',input_tensor, input_MPS[site_y].conj(), self.H[site_y], input_MPS[site_y])
        return output_tensor

    def _cal_left_tensor(self, input_MPS, site_x):
        """calcuate the rank-3 tensor on the left of the effective H"""
        for site_i in range(site_x):
            if site_i == 0:
                output_tensor = np.einsum('aj,abn,bl->jnl',np.squeeze(input_MPS[site_i].conj()), np.squeeze(self.H[site_i]), np.squeeze(input_MPS[site_i]))
            else:
                output_tensor = self._contract(input_MPS, output_tensor, site_i)
        return output_tensor

    def _cal_right_tensor(self, input_MPS, site_x):
        """calcuate the rank-3 tensor on the right of the effective H"""
        L = self.L
        for i in range(L-site_x):
            site_i = L - i - 1
            if site_i == L - 1:
                output_tensor = np.einsum('ia,mab,kb->imk',np.squeeze(input_MPS[site_i].conj()), np.squeeze(self.H[site_i]), np.squeeze(input_MPS[site_i]))
                 # base case
            else:
                output_tensor = self._contract(input_MPS, output_tensor, site_i, right=True)
        return output_tensor

    def _cal_eff_H(self, input_MPS, site):
        """calculate effective one-site local Hamiltonian"""
        L = self.L
        if site != 0 and site != L-1:
            left_tensor = self._cal_left_tensor(input_MPS, site)
            right_tensor = self._cal_right_tensor(input_MPS, site+1)

            dim = left_tensor.shape[0]*self.H[site].shape[1]*right_tensor.shape[0]

            H_eff = np.einsum('ijk,jabm,lmn->ialkbn', left_tensor, self.H[site], right_tensor).reshape(dim, dim)

        # deal with edge cases
        elif site == 0:
            right_tensor = self._cal_right_tensor(input_MPS, site+1)
            dim = self.H[site].shape[1] * right_tensor.shape[0]
            H_eff = np.einsum('abm,lmn->albn', np.squeeze(self.H[site]), right_tensor).reshape(dim, dim)

        else:
            left_tensor = self._cal_left_tensor(input_MPS, site)
            dim = self.H[site].shape[1] * left_tensor.shape[0]
            H_eff = np.einsum('mab,lmn->albn', np.squeeze(self.H[site]), left_tensor).reshape(dim, dim)

        return H_eff

    def _cal_eff_K(self, input_MPS, site):
        """calculate effective zero-site Hamiltonian"""
        L = self.L
        if site != L-1:
            left_tensor = self._cal_left_tensor(input_MPS, site+1)
            right_tensor = self._cal_right_tensor(input_MPS, site+1)

            dim = left_tensor.shape[0]*right_tensor.shape[0]

            K_eff = np.einsum('ijk,ljn->ilkn', left_tensor, right_tensor).reshape(dim, dim)

        else:
            pass
            # left_tensor = self._cal_left_tensor(input_MPS, site+1)
            # dim = left_tensor.shape[0]
            # K_eff = np.squeeze(left_tensor).reshape(dim, dim)

        return K_eff

    def ground_state_search(self, num_sweep=10, D=5):
        """implement the ground state search alogorithm that iteratively optimize the MPS site by site"""
        L = self.L
        # Step 1: initialize a random MPS
        trial_MPS = self._initialize_mps(D)
        # Step 2: bring the initial MPS into a right normalize form
        # trial_MPS = self._left_canonical(trial_MPS, D)
        trial_MPS = self._right_canonical(trial_MPS, D)
        # for i in range(L):
            # print(f"intial MPS {i+1} shape {trial_MPS[i].shape}")
        # define a python dictionary store energy data
        energy_dic = {
        "sweep":[],
        "iteration":[],
        "energy":[],
        "energy variance": []
         }

        # loop over each site and sweep back and force
        for iteration in range(num_sweep):
            for i in range(L):
                if iteration%2 == 0:
                    site = i # sweep from left to right
                    right_sweep = True
                else:
                    site = self.L - i - 1 # sweep from right to left
                    right_sweep = False

                # skip the first site to avoid repeating optimization of the same site
                if iteration == 0 or (i!=0 and iteration != 0):

                    # step 3: calcuate effection rank-6 effection Hamiltonian on each site
                    H_eff = self._cal_eff_H(trial_MPS, site)
                    # print(f'H_eff:\n{H_eff}')
                    # assert np.allclose(H_eff, H_eff.transpose().conj())

                    # step 4: diagonalize the Hamiltonian
                    E, V = np.linalg.eigh(H_eff)
                    # sort eigenvalue and eigenvectors
                    idx = E.argsort()
                    E = E[idx]
                    V = V[:,idx]

                    # step 5: update the local MPS
                    left_bond_dim, phys_dim, right_bond_dim = trial_MPS[site].shape
                    optimized_tensor = V[:,0].reshape(left_bond_dim, phys_dim, right_bond_dim)
                else:
                    optimized_tensor = trial_MPS[site]

                # left normalize the optimized tensor if right sweep
                if right_sweep:
                    optimized_tensor = optimized_tensor.reshape(left_bond_dim*phys_dim, right_bond_dim)
                    A, S, Vh = np.linalg.svd(optimized_tensor, full_matrices=False)
                    # change right bond dimension for base cases
                    right_bond_dim = min(left_bond_dim*phys_dim, right_bond_dim)

                    trial_MPS[site] = A.reshape(left_bond_dim, phys_dim, right_bond_dim)
                    if site < L-1:
                        # print(f'site:{site+1}')
                        # print(S.shape)
                        # print(V.shape)
                        # print(trial_MPS[site+1].shape)
                        trial_MPS[site+1] = np.einsum('s,sa,aib->sib', S, Vh, trial_MPS[site+1]).copy()
                    else:
                        pass

                # right normalize the optimized tensor if left sweep
                else:
                    optimized_tensor=optimized_tensor.reshape(left_bond_dim, right_bond_dim*phys_dim)
                    U, S, B = np.linalg.svd(optimized_tensor, full_matrices=False)
                    left_bond_dim = min(phys_dim*right_bond_dim, left_bond_dim)
                    trial_MPS[site] = B.reshape(left_bond_dim, phys_dim, right_bond_dim)
                    if site > 0:
                        trial_MPS[site-1] = np.einsum('aib,bs,s->ais', trial_MPS[site-1], U, S).copy()
                    else:
                        pass

                # Step 6: calcuate energy variance w.r.t the trial MPS
                variance = self._cal_variance(self.H, trial_MPS)

                print(f"Sweep {iteration}, site {site}: energy {E[0]}, variance: {variance.real}")

                # store the data
                energy_dic["sweep"].append(iteration+1)
                energy_dic["iteration"].append(L*iteration+i)
                energy_dic['energy'].append(E[0])
                energy_dic['energy variance'].append(variance.real)


        df = pd.DataFrame(energy_dic)
        df.to_csv("spin_Hamiltonain_DMRG_GS_search_data.csv", index=False)

    def TDVP_evolution(self, t_final, num_sweep, D, imagine_t):
        """1-site TDVP time evolution, following the algorithm at
        https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html

        Each full period (one right sweep + one left sweep) advances the
        state by delta_t: every site is forward-evolved by delta_t/2 in
        each sweep (no special-cased boundary "full step"), and every bond
        is backward-evolved by delta_t/2 in each sweep. Left/right
        environments (L_env, R_env) are cached and updated incrementally
        rather than recomputed from scratch at each site.
        """
        L = self.L
        delta_t = t_final / num_sweep
        if imagine_t:
            delta_t *= (-1j)

        trial_MPS = self._initialize_mps(D)
        trial_MPS = self._right_canonical(trial_MPS, D)

        energy_dic = {
            "time": [],
            "energy expectation value": [],
            "total magnetization": []
        }

        trivial_env = np.ones((1, 1, 1), dtype=complex)

        def contract_left(Lenv, A, site):
            """L_j = CONTRACT-LEFT(L_{j-1}, W_j, A_j)"""
            return np.einsum('imk,iaj,mabn,kbl->jnl', Lenv, A.conj(), self.H[site], A)

        def contract_right(Renv, B, site):
            """R_j = CONTRACT-RIGHT(R_{j+1}, W_j, B_j)"""
            return np.einsum('jnl,iaj,mabn,kbl->imk', Renv, B.conj(), self.H[site], B)

        def build_Heff(Lenv, site, Renv):
            """H_eff = L_{j-1} . W_j . R_{j+1}, reshaped to match M.ravel() order"""
            p = Lenv.shape[0]
            r = Renv.shape[0]
            W = self.H[site]
            tensor = np.einsum('pmq,mabn,rns->parqbs', Lenv, W, Renv)
            dim = p * W.shape[1] * r
            return tensor.reshape(dim, dim)

        def build_Keff(Lenv, Renv):
            """K_eff = L_j . R_{j+1} (no local operator), reshaped to match C.ravel() order"""
            p = Lenv.shape[0]
            r = Renv.shape[0]
            tensor = np.einsum('pmq,rms->prqs', Lenv, Renv)
            dim = p * r
            return tensor.reshape(dim, dim)

        def expm_apply(H_matrix, vec, coeff):
            """apply exp(-1j * H_matrix * coeff) to vec"""
            E, V = np.linalg.eigh(H_matrix)
            expH = V @ np.diag(np.exp(-1j * E * coeff)) @ V.conj().T
            return expH @ vec

        def record(trial_MPS, time):
            energy_exp = self._cal_expectation(trial_MPS, self.H)
            magnetization = self._cal_expectation(trial_MPS, self.Sz_total_MPO)
            print('time step: {:.4f} , energy: {:.4f}, magnetization: {:.4f}'.format(
                time, energy_exp, magnetization.real))
            energy_dic['time'].append(time)
            energy_dic['energy expectation value'].append(energy_exp.real)
            energy_dic['total magnetization'].append(magnetization.real)

        # precompute the initial right environments (state starts right-canonical)
        R_env = {L: trivial_env}
        for site in range(L - 1, -1, -1):
            R_env[site] = contract_right(R_env[site + 1], trial_MPS[site], site)

        L_env = {-1: trivial_env}

        for period in range(num_sweep):
            record(trial_MPS, period * delta_t)

            # ---- SWEEP RIGHT (updates L_env, uses the static R_env) ----
            L_env = {-1: trivial_env}
            for site in range(L):
                Lenv = L_env[site - 1]
                Renv = R_env[site + 1]

                # (a) evolve A_C(site) forward by delta_t/2
                shape = trial_MPS[site].shape
                H_eff = build_Heff(Lenv, site, Renv)
                trial_MPS[site] = expm_apply(H_eff, trial_MPS[site].ravel(), delta_t / 2).reshape(shape)

                # (b) QR decompose into left-orthonormal A_site and bond matrix C
                left_bond_dim, phys_dim, right_bond_dim = shape
                Q, C = np.linalg.qr(trial_MPS[site].reshape(left_bond_dim * phys_dim, right_bond_dim))
                k = Q.shape[1]
                trial_MPS[site] = Q.reshape(left_bond_dim, phys_dim, k)

                # (c) update the left environment with the new isometry
                L_env[site] = contract_left(Lenv, trial_MPS[site], site)

                if site != L - 1:
                    # (d) evolve the bond matrix backward by delta_t/2
                    K_eff = build_Keff(L_env[site], Renv)
                    C = expm_apply(K_eff, C.ravel(), -delta_t / 2).reshape(C.shape)
                    # (e) absorb C into the next site
                    trial_MPS[site + 1] = np.einsum('ac,cib->aib', C, trial_MPS[site + 1])

            # ---- SWEEP LEFT (updates R_env, uses the static L_env) ----
            R_env = {L: trivial_env}
            for site in range(L - 1, -1, -1):
                Lenv = L_env[site - 1]
                Renv = R_env[site + 1]

                # (a) evolve A_C(site) forward by delta_t/2
                shape = trial_MPS[site].shape
                H_eff = build_Heff(Lenv, site, Renv)
                trial_MPS[site] = expm_apply(H_eff, trial_MPS[site].ravel(), delta_t / 2).reshape(shape)

                # (b) QR decompose (from the right) into right-orthonormal B_site and bond matrix C
                left_bond_dim, phys_dim, right_bond_dim = shape
                Q, R_factor = np.linalg.qr(trial_MPS[site].reshape(left_bond_dim, phys_dim * right_bond_dim).T)
                k = Q.shape[1]
                trial_MPS[site] = Q.T.reshape(k, phys_dim, right_bond_dim)
                C = R_factor.T  # shape (left_bond_dim, k)

                # (c) update the right environment with the new isometry
                R_env[site] = contract_right(Renv, trial_MPS[site], site)

                if site != 0:
                    # (d) evolve the bond matrix backward by delta_t/2
                    K_eff = build_Keff(L_env[site - 1], R_env[site])
                    C = expm_apply(K_eff, C.ravel(), -delta_t / 2).reshape(C.shape)
                    # (e) absorb C into the previous site
                    trial_MPS[site - 1] = np.einsum('aib,bc->aic', trial_MPS[site - 1], C)

        record(trial_MPS, num_sweep * delta_t)

        # store data
        df = pd.DataFrame(energy_dic)
        df.to_csv("TDVP_energy_data.csv", index=False)

        return
