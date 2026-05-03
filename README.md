# UniformTEMPO.jl
Welcome to _UniformTEMPO.jl_, a Julia implementation of the open quantum system algorithm introduced in Ref. [[Link, Tu, Strunz, PRL 123 200403 (2024)](https://doi.org/10.1103/PhysRevLett.132.200403)]. It uses a representation of __open quantum system dynamics__ with __arbitrary stationary Gaussian baths__ in terms of infinite tensor networks. Specifically, a process tensor representing the system-bath interaction is generated in a uniform matrix product operator form using infinite time evolving block decimation. The process tensor can then be used to compute multi-time dynamics of open quantum systems. 

## Documentation

The documentation is available [here](https://uniformtempo.github.io/UniformTEMPO.jl/).

## Installation

This package is not registered as an official package. To install it directly from github, use

```julia
julia> ]
(myenv) pkg> add https://github.com/uniformTEMPO/UniformTEMPO.jl.git
```


## Basic usage

The code does not introduce abstract types for quantum operators and states. States and operators are represented by basic Julia arrays. We aim to avoid abstraction in favor of code simplicity and readability.

Consider an open system model with a Gaussian bosonic bath, such as

$$H = H_\mathrm{sys}\otimes\mathbb{1}_\mathrm{env} + S\otimes \sum_\lambda g_\lambda (b_\lambda+b_\lambda^\dagger)+\mathbb{1}_\mathrm{sys}\otimes\sum_\lambda \omega_\lambda b_\lambda^\dagger b_\lambda $$

The effect of the bath on the system is characterized by the bath correlation function

$$ \mathrm{bcf}(t)=\sum_\lambda g_\lambda^2 \big(\coth(\beta\omega_\lambda/2)\cos(\omega_\lambda t)-\mathrm{i}\sin(\omega_\lambda t)\big)
$$

We also provide an implementation of uniTEMPO for general linear coupling models based on Ref. [[Link, arXiv:2603.23432 (2026)](https://arxiv.org/abs/2603.23432)].

Here is a simple usage example for the spin boson model.

(1) Provide a bath correlation function and a coupling operator

```julia
bcf(t) = 0.1 * (5 / (1 + im * 5 * t))^2
S = [0 1; 1 0]
```

(2) compute a process tensor MPO using uniTEMPO, 

```julia
using UniformTEMPO
pt = uniTEMPO(S, 0.1, bcf, 1e-8)
```

(3) use it to compute dynamics

```julia
ρ0 = [1 0; 0 0]
ρt = evolve(pt, ρ0, 1000; h_s=[1 0; 0 -1]) # evolve 1000 steps
ρt[end] # system state at the end of the evolution
```

Further descriptions are provided in the documentation. Also check out the various examples provided under `./examples`.

## Citation

If you use this code in a publication, please cite the corresponding references listed in the docs.