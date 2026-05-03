# Advanced

## Obtaining the full system and auxiliary state

Sometimes it is necessary to recover the full computational state of system and auxiliary degrees of freedom after time evolution. This can be achieved generally with the keyword argument `return_full=true`. 

```julia
N = 100 # number of time steps
xN = evolve(pt, ρ0 ,N; return_full=true)
size(xN) == (bond_dim(pt), pt.s_dim^2) # true
```

Note that the full state `xN` can be used as an initial state for further evolution

```julia
M = 100 # number of time steps
xNM = evolve(pt, xN, M; return_full=true) # computational state after N+M steps
```

## Combining two environments

When a system is coupled to two independent evironments, their process tensors can be combined via the `+` operator. For instance a single spin coupled to two independent reserviors via $\sigma_x$ and $\sigma_z$ respectively. We first create the two individual process tensors

```julia
σ_x = [0 1; 1 0]
σ_z = [1 0; 0 -1]
delta_t = 0.05
bcf(t) = 0.2 * exp(-abs(t) - im * t) # a simple bcf
pt1 = uniTEMPO(σ_x, delta_t, bcf, 1e-8)
pt2 = uniTEMPO(σ_z, delta_t, bcf, 1e-8)
bond_dim(pt1) # 10
```

Then we combine the process tensors via the ACE combination [[Cygorek et.al., Nature Physics 18, 662-668 (2022)](https://www.nature.com/articles/s41567-022-01544-9)]

```julia
pt = pt1 + pt2
bond_dim(pt) # 100
```

The new bond dimension is the product of the individual bond dimensions. To truncate the bond dimension using a truncation scheme based on iMPS compression [[Parker, Cao, Zaletel, PRB 102, 035147 (2020)](https://doi.org/10.1103/PhysRevB.102.035147)] use [`truncate`](@ref UniformTEMPO.truncate)

```julia
pt_c = UniformTEMPO.truncate(pt, 1e-8) # truncate to lower bond dimension
bond_dim(pt_c) # 25
```

## Multiple coupling operators

The function [`uniTEMPO`](@ref uniTEMPO) also directly supports multiple coupling operators via the scheme from Ref. [[Link, arXiv.2603.23432 (2026)](https://arxiv.org/abs/2603.23432)]. Then the bath correlation function is matrix-valued. The syntax is simply

```julia
bcf_mat(t) = [1 0; 0 1] * bcf(t)
pt = uniTEMPO([σ_x, σ_z], delta_t, bcf_mat, 1e-8)
```

## ITensors export

To export a `UniformPTMPO` to `ITensors.jl`, load `ITensors` which enables the extension. Then export via

```julia
using ITensors
pti = to_ITensor(pt)
```

To obtain the propagator in Hilbert-space rather than Liouville-space, use the combiner `pti.cmb`

```julia
q = pti.q * pti.cmb * pti.cmb'
inds(q) # ((dim=31|id=202|"AuxBathSite")', (dim=31|id=202|"AuxBathSite"), (dim=2|id=991|"Site+"), (dim=2|id=305|"Site-"), (dim=2|id=991|"Site+")', (dim=2|id=305|"Site-")')
```


## Notes on performance

The code is optimized for performance and relies mostly on basic linear algebra routines. Some functions also use TensorOperations.jl for einsum-like tensor contractions. All functions take abstract array types so the code is mostly generic. On Intel or AMD x86 hardware, using Intel MKL linear algebra routines gives a significant speedup. Just load `MKL` first to switch the backend

```julia
julia> using MKL
julia> using UniformTEMPO
```

If the system dimension is large it is generally recommended to use a low-rank singular value decomposition in uniTEMPO

```julia
uniTEMPO(S, delta_t, bcf, tol; low_rank_svd=true)
```

and/or to activate the filtering algorithm from Ref. [[Cochin et.al., arXiv:2603.06840 (2026)](https://doi.org/10.48550/arXiv.2603.06840)] by specifying a non-zero filtering tolerance

```julia
uniTEMPO(S, delta_t, bcf, tol; svd_filtering_tol=1e-8)
```
