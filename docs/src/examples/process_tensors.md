# Process tensors

General quantum process tensors are multi-linear maps from a set of "interventions" (system channels) at a given array of times $t_1<\ldots < t_{n-1}$ to the system state at time $t_n$, see Ref. [[Pollock et.al., PRA 97, 012127 (2018)](https://doi.org/10.1103/PhysRevA.97.012127)]. We use superoperator formalism and the following convention for process tensors
```math
T[(μ^1_I,μ^1_O),...,(μ^{n-1}_I,μ^{n-1}_O),μ^n]
```
where $I$/$O$ stands for input and output index. These indices can be contracted with the input and output indices of a channel in superoperator form. The last index is the index of the vectorized output density matrix. One can also define a process tensor for the case that no system initial state is provided (intervention at $t_0=0$). Then the process tensor has an additional index
```math
T[μ^0,(μ^1_I,μ^1_O),...,(μ^{n-1}_I,μ^{n-1}_O),μ^n]
```
that can be contracted with the system initial state. Process tensors are also often written in their choi representation. To convert a process tensor to its corresponding choi state, use the function [`pt_to_choi`](@ref UniformTEMPO.pt_to_choi)

Process tensors can be computed directly from a PT-MPO using the function [`process_tensor`](@ref process_tensor). As an example we compute the process tensor for a single intervention for the spin boson model.

We first compute the PT-MPO via [`uniTEMPO`](@ref uniTEMPO), here for a simple Lorentzian bath
```julia
using UniformTEMPO
pt = uniTEMPO([0 1; 1 0], 0.1, t -> exp(-0.5 * abs(t) - im * t), 1e-8)
h_sys = [1 0; 0 -1]
```
We consider quench dynamics for $t_2$ time steps with an intervention after $t_1$ time steps
```julia
t_1 = 10; t_2 = 20
proc = process_tensor(pt, [t_1, t_2]; h_s=h_sys)
```
To check if its a valid process tensor, we compute the choi state
```julia
χ = pt_to_choi(proc)
```
This will have dimensions $(2\times 4\times 2)\times(2\times 4\times 2)$, corresponding to the initial state, the intervention, and the final state. The choi state is a density matrix (up to uniTEMPO compression errors)
```julia
using LinearAlgebra
pχ = eigvals(reshape(χ, 16, 16))
sum(pχ) # 0.9999936653183381 + 7.550981339638109e-15im
```
One can also compute stationary process tensors, where the initial state is the correlated steady-state of system and bath. For this we first compute the full steady state (see [advanced](../advanced.md)) and then plug this in as an initial state for the `process_tensor` function
```julia
x0 = steadystate(pt; h_s=h_sys, return_full=true)
proc = process_tensor(pt, x0, [0, t_2 - t_2]; h_s=h_sys)
χ = pt_to_choi(proc)
```
The choi state will then have dimension $(4\times 2)\times(4\times 2)$ corresponding to intervention and output state.

Process tensors can be used to characterize temporal correlations of the dynamics in a general multi-time framework, including Markovianity and quantum memory. This code was used in Ref. [[Bäcker, Link, Strunz, arXiv:2505.13067 (2025)](https://doi.org/10.48550/arXiv.2505.13067
)] for computing pocess tensors for spin boson models.