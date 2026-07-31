
this repo is a playground the test basic ideas of DMRG/tensor networks

## TDVP time evolution

`spin_Hamiltonian.TDVP_evolution` (in `spin_Hamiltonian/spin_Hamiltonian.py`) implements
1-site TDVP (Time-Dependent Variational Principle) real/imaginary time evolution for a
spin chain MPS, following the algorithm described at
[tensornetwork.org/mps/algorithms/timeevo/tdvp](https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html).

### Algorithm

Each call evolves the state for a total time `t_final`, split into `num_sweep` periods of
length

$$
\delta t = \frac{t_{\text{final}}}{\text{num \ sweep}},
$$

and, when `imagine_t=True`, $\delta t \to -i\,\delta t$ (imaginary time, used to relax
towards the ground state instead of propagating in real time).

The full time-dependent Schrödinger equation $i\,\partial_t |\Psi\rangle = H|\Psi\rangle$
is not integrable exactly on the manifold of MPS with fixed bond dimension $D$. TDVP
instead projects the right-hand side onto the tangent space of that manifold at the
current state, $i\,\partial_t|\Psi\rangle = P_{T_\Psi\mathcal{M}} H |\Psi\rangle$, and the
1-site algorithm solves this projected equation by sweeping through the chain and
splitting each bond's contribution into a **site update** and a **bond update**
(the "projector-splitting" integrator). Every period performs one right sweep followed by
one left sweep across the chain; each sweep advances the state by $\delta t/2$, so a full
period advances it by $\delta t$.

For each site visited during a sweep:

#### 1. Forward-evolve the site tensor
   
The one-site reduced/effective Hamiltonian is built
from the left and right environment tensors $L$, $R$ and the site's MPO tensor $W$,

$$
(H_{\text{eff}})_{(p,a,r),(q,b,s)} = \sum_{m,n} L_{p m q}\; W_{m a b n}\; R_{r n s},
$$

   and the site tensor $A_C(\text{site})$ (flattened over its left-bond/physical/right-bond
   indices) is propagated forward by $\delta t/2$:

$$
A_C(\text{site})  \leftarrow \exp\big(-i\,H_{\text{eff}}\,\delta t/2\big)\;A_C(\text{site}).
$$

#### 2. Orthogonalize the evolved tensor via QR decomposition,

   $$
   A_C(\text{site}) = Q\,C \quad(\text{sweeping right, } Q \text{ left-orthonormal}),
   \qquad
   A_C(\text{site}) = C\,Q \quad(\text{sweeping left, } Q \text{ right-orthonormal}),
   $$

which factors out an orthonormal site tensor $Q$ (stored in place of $$A_C(\text{site})$$)
and a bond matrix $C$.

#### 3. Update the environment (`L_env` when sweeping right, `R_env` when sweeping left)

   incrementally using the newly orthogonalized site tensor $Q$, e.g. for a right sweep

   $$
   L_{j}[j',n,l] = \sum_{i,m,k,a,b} L_{j-1}[i,m,k]\;Q^{*}_j[i,a,j']\;W_j[m,a,b,n]\;Q_j[k,b,l],
   $$

   rather than recomputing $L$ (or $R$) from scratch at every site.

#### 4. Backward-evolve the bond matrix: The zero-site (bond) effective Hamiltonian omits

   the local MPO operator entirely,

   $$
   (K_{\text{eff}})_{(p\,r),(q\,s)} = \sum_{m} L_{p m q}\; R_{r m s},
   $$

   and the bond matrix $C$ is propagated **backward** by $\delta t/2$ — i.e. with the
   opposite sign convention from step 1 — to compensate for the extra evolution the site
   update injected at that bond:

   $$
   C \leftarrow \exp\big(+i\,K_{\text{eff}}\,\delta t/2\big)\;C,
   $$

   after which $C$ is absorbed into the neighboring (not-yet-visited) site tensor.

Steps 1–4 realize the symmetric Lie–Trotter splitting of the tangent-space projector into
alternating single-site ($H_{\text{eff}}$, forward half-step) and single-bond
($K_{\text{eff}}$, backward half-step) propagators. This is the defining feature of 1-site
TDVP: because every propagator $$\exp(-iH_{\text{eff}}\tau)$$ / $$\exp(-iK_{\text{eff}}\tau)$$ is
unitary, the MPS norm is conserved exactly and the bond dimension $D$ never grows, unlike
Trotter-Suzuki gate application followed by truncation.

Before the first sweep the initial right environments $R_j$ are precomputed once via the
recursion above (the initial MPS starts in right-canonical form, so $L_{-1}$ and $R_L$ are
both the trivial $1\times1\times1$ tensor equal to $1$); after that, `L_env`/`R_env` are
cached dictionaries updated site-by-site rather than being rebuilt from scratch every step.

### Observables

At the start of every period, the energy expectation value `<H>` and the (optionally
chemical-shift-weighted) total magnetization `<sum_i chemical_shift[i] * S_z(i)>` are
recorded via `_cal_expectation` against `self.H` and `self.Sz_total_MPO` respectively.
Results are collected into a dataframe and written to `TDVP_energy_data.csv`.

### Usage

```python
from spin_Hamiltonian import spin_Hamiltonian

L, J, Jz, h = 10, 1.0, 1.0, 1.0
model = spin_Hamiltonian(L, J, Jz, h)  # optional chemical_shift=array of length L

# real-time evolution: t_final total time, split into num_sweep periods, bond dim D
model.TDVP_evolution(t_final=10, num_sweep=200, D=4, imagine_t=False)

# imaginary-time evolution (relaxation towards the ground state)
model.TDVP_evolution(t_final=10, num_sweep=200, D=4, imagine_t=True)
```
