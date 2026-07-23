# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a personal playground for prototyping tensor-network / DMRG (Density Matrix Renormalization Group) algorithms in plain NumPy — no tensor-network library (ITensor, TeNPy, quimb, etc.) is used. Every algorithm is implemented from scratch with `np.einsum` and `np.linalg.svd` so the underlying linear algebra stays visible.

## Commands

There is no build system, package manifest, linter, or test runner (no `requirements.txt`, `setup.py`, `pytest`, etc.). Dependencies are plain third-party imports that must already be installed in the environment: `numpy`, `scipy`, `pandas`, `matplotlib`, `parse`, `opt_einsum`.

Each top-level module directory is runnable independently via its `test.py`, which is a manual demo script (prints shapes/values and exercises the algorithm) rather than a pytest/unittest suite — there are no test assertions run via a test framework:

```bash
python3 MPS_decomposition/test.py
python3 MPO_compression/test.py
python3 spin_Hamiltonian/test.py
```

Correctness is instead self-checked *inside* the algorithms via `assert np.allclose(...)` statements (e.g. verifying canonical-form orthogonality, or that a decomposition reconstructs the original tensor). When editing an algorithm, keep these in-line asserts intact/passing rather than adding an external test framework.

**Known quirk:** every `test.py` hardcodes an absolute `sys.path.insert` to `/Users/pauliebao/Tensor_play_ground/<module>/` to import its sibling module. If you move the repo or add a new module's `test.py`, follow the same pattern used by the existing files (see any `test.py` for the template) rather than introducing relative imports elsewhere.

`spin_Hamiltonian/Plot_DMRG.ipynb` reads the CSV output of the ground-state search (see below) to plot convergence.

## Architecture

### Data representation convention

- An MPS/MPO is a Python **dict keyed by site index** (0 to L-1), not a list — sites are frequently accessed out of order (e.g. sweeping right-to-left) so this is intentional, not incidental.
- **MPS tensor shape:** `(left_bond_dim, phys_dim, right_bond_dim)`. Edge sites (site 0 and site L-1) carry a dummy bond dimension of 1.
- **MPO tensor shape:** `(left_bond_dim, phys_dim_out, phys_dim_in, right_bond_dim)`.
- Contractions are written as explicit `np.einsum` index strings; get familiar with a module's einsum subscripts before modifying it — the shape/leg semantics live in those strings, not in comments.

### Module dependency graph

Modules build on each other via subclassing, reusing the parent's random-tensor constructor and canonicalization routines rather than duplicating them:

```
MPS_canonical (MPS_canonical.py)         — base: builds a random MPS, left_canonical()/right_canonical()
├── MPO_contraction (MPO_contraction.py) — adds a random MPO, contracts MPO·MPS
└── MPS_compression (MPS_compression.py)— adds SVD_compress() and variational iterative_compress()
    └── MPO_compression (MPO_compression.py) — flattens MPO's two physical legs into one, reuses MPS compression, restores legs

MPS_decompose (MPS_decompose.py)         — base: decomposes a random high-rank tensor into MPS (left_decompose()/right_decompose() via sweeping SVD)
└── MPO_decomposite (MPO_decomposite.py) — flattens two physical dims (D1*D2) into one before decomposing, restores them after

MPS_contraction (MPS_contraction.py)     — standalone: builds its own MPS + local operators, computes expectation values
MPO_multiply (MPO_multiply.py)           — standalone: builds two random MPOs, contracts their product site-by-site

spin_Hamiltonian (spin_Hamiltonian.py)   — standalone (does not subclass the above, despite reimplementing similar
                                            left/right canonicalization internally)
```

When adding a new algorithm that needs canonical-form MPS or an existing compression/decomposition primitive, prefer subclassing the relevant existing class (as the modules above do) over reimplementing it, unless there's a reason to keep it standalone as `spin_Hamiltonian` does.

### `spin_Hamiltonian`: variational MPS (VMPS) / single-site DMRG ground-state search

This is the most involved module — it implements the variational matrix-product-state ground-state search (equivalent to single-site DMRG). The theory below maps directly onto the code so that changes can be checked against the underlying math.

#### 1. The Hamiltonian as an MPO

The spin-1/2 XXZ chain

```
H = Σ_i  [ -J/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + Jz Sz_i Sz_{i+1} - h Sz_i ]
```

is written as a matrix-product operator with bulk bond dimension 5, following Schollwöck (*Annals of Physics* 326 (2011), Eq. 182 / chapter 6). Each bulk tensor `W[site]` has shape `(5, 2, 2, 5)` and is built from a triangular "transfer matrix" of operators so that contracting `W[0] · W[1] · ... · W[L-1]` (the boundary tensors being the first row / last column of that structure) reproduces the full sum over bonds:

```
        ⎡  I     0     0     0    0 ⎤
        ⎢  S+    0     0     0    0 ⎥
W[i] =  ⎢  S-    0     0     0    0 ⎥      (rows = "left" MPO bond, cols = "right" MPO bond)
        ⎢  Sz    0     0     0    0 ⎥
        ⎣ -h·Sz -J/2·S- -J/2·S+ Jz·Sz  I ⎦
```

`self.H[site]` in the code *is* this tensor; the first site keeps only the bottom row and the last site keeps only the left column (open boundary conditions ⇒ dummy bond dimension 1 at the edges), matching the `left_bond_dim`/`right_bond_dim` logic in `__init__`.

#### 2. MPS ansatz and the variational principle

The trial wavefunction is a matrix-product state of fixed bond dimension `D`:

```
|ψ⟩ = Σ_{s1...sL}  A[0]^{s1} A[1]^{s2} ... A[L-1]^{sL}  |s1 s2 ... sL⟩
```

where `A[site]` has shape `(left_bond_dim, phys_dim=2, right_bond_dim)` — exactly `_initialize_mps`'s random tensor. VMPS finds the ground state by minimizing the Rayleigh quotient over all such MPS of bond dimension `D`:

```
E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
```

This is a nonlinear optimization in the full set of tensors `{A[0], ..., A[L-1]}` simultaneously, but it becomes a simple *linear* eigenvalue problem one site at a time if the MPS is kept in **mixed canonical form**.

#### 3. Canonical form makes the norm trivial

A tensor is **left-canonical** if `Σ_s A^{s†} A^s = I` (contracting over the left bond and physical index) and **right-canonical** if `Σ_s B^s B^{s†} = I` (contracting over the right bond and physical index). These are exactly the conditions asserted in `_left_canonical`/`_right_canonical` and in `MPS_canonical.py`:

```python
assert np.allclose(np.einsum('bia,bic->ac', tensor, tensor), np.eye(right_bond_dim))   # left-canonical
assert np.allclose(np.einsum('aib,cib->ac', tensor, tensor), np.eye(left_bond_dim))    # right-canonical
```

If every site to the left of site `l` is left-canonical and every site to the right of `l` is right-canonical (an MPS in this state has its **orthogonality center** at site `l`), then all of those tensors telescope into identity contractions and

```
⟨ψ|ψ⟩ = ⟨A[l] | A[l]⟩        (just the local tensor's self-overlap — no other sites involved)
```

This is *why* the algorithm re-canonicalizes the just-updated tensor via SVD after every local update (the `right_sweep`/else branches at the end of the site loop in `ground_state_search`) before moving to the next site: it keeps exactly one site — the next one to be optimized — un-canonicalized, i.e. shifts the orthogonality center by one site each step.

#### 4. The local effective Hamiltonian

With the norm reduced to the identity, minimizing `E[ψ]` over the single tensor `A[l]` (holding all others fixed) is minimizing `⟨A[l]| H_eff |A[l]⟩ / ⟨A[l]|A[l]⟩`, i.e. an ordinary Rayleigh quotient, where `H_eff` is the Hamiltonian "projected" into the current tensor's index space. Diagrammatically, `H_eff` is built by contracting:

- a left environment tensor `L` = everything to the left of site `l` (MPS bra, `H`, MPS ket, contracted down to a single rank-3 tensor `(bond_bra, mpo_bond, bond_ket)`),
- the local MPO tensor `W[l]`,
- a right environment tensor `R` = everything to the right of site `l`, built the mirror-image way,

into one big matrix over the combined `(left_bond, phys, right_bond)` index of site `l`. This is precisely `_cal_left_tensor`/`_cal_right_tensor`/`_cal_eff_H`:

```python
H_eff = einsum('ijk,jabm,lmn->ialkbn', L, W[l], R).reshape(dim, dim)   # dim = left_bond * phys_dim * right_bond
```

`L` and `R` are built incrementally site-by-site (`_contract`), which is the standard DMRG trick of updating environment tensors rather than recomputing the full contraction from scratch at every site — an O(L) saving per sweep step.

Because the environment (everything outside site `l`) is exactly canonical/orthonormal, the generalized eigenvalue problem `H_eff v = E · N v` collapses to the *ordinary* Hermitian eigenvalue problem

```
H_eff v = E v
```

which is why the code calls `np.linalg.eigh(H_eff)` directly (after asserting `H_eff` is Hermitian) and takes the lowest eigenpair `(E[0], V[:,0])` as the new, energy-minimizing local tensor — reshaped back to `(left_bond_dim, phys_dim, right_bond_dim)`.

#### 5. The sweep

`ground_state_search(num_sweep, D)` repeats the following for `num_sweep` full passes over the chain, alternating direction each pass (`iteration % 2`, matching standard DMRG sweeping):

1. Build `H_eff` at the current site from the (already-canonical) environment.
2. Diagonalize it; take the ground eigenvector as the new local tensor.
3. SVD the new local tensor to push the orthogonality center one site over in the sweep direction (`right_sweep` branch: reshape as `(left·phys, right)`, keep `U` (left-canonical) at this site, absorb `S·Vh` into the next site to the right; the `else` branch mirrors this leftward using `(left, phys·right)` and `B`/right-canonical).
4. Move to the neighboring site and repeat.

The very first site touched in each direction is skipped from re-optimization (`if iteration == 0 or (i != 0 and iteration != 0)`) because it was already the last site optimized at the end of the previous sweep leg — re-solving it immediately would be redundant.

This is **single-site** DMRG: the bond dimension `D` is fixed for the whole run (passed in once) rather than grown adaptively from a two-site update, so it cannot discover a larger bond dimension than the initial random MPS was given, and (as in real single-site DMRG) it can in principle get stuck without noise/subspace-expansion terms — there's no such regularization implemented here.

#### 6. Convergence diagnostic: energy variance

Since an exact eigenstate satisfies `H|ψ⟩ = E|ψ⟩`, its variance `⟨H²⟩ − ⟨H⟩²` is exactly zero; a converged VMPS run should drive this to ~0. `_cal_variance` computes `⟨H²⟩` by first forming the MPO-square `H·H` site-by-site (`_cal_MPO_MPO_product`, an `iabj,kbcl->ikacjl` contraction that multiplies bond dimensions, i.e. 5×5=25 per bulk site) and then taking its expectation value (`_cal_expectation`, the sandwich `bra · MPO · ket` contracted site-by-site) against the current trial MPS. Both `energy` and `energy variance` are logged per sweep step and written to `spin_Hamiltonain_DMRG_GS_search_data.csv` (note the existing filename typo — match it if reading/writing this file), which `Plot_DMRG.ipynb` visualizes.
