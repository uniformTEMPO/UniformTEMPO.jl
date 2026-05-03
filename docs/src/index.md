# UniformTEMPO.jl

## Welcome

Welcome to the documentation for _UniformTEMPO.jl_, a Julia implementation of the open quantum system algorithm introduced in Ref. [[Link, Tu, Strunz, PRL 123 200403 (2024)](https://doi.org/10.1103/PhysRevLett.132.200403)]. It uses a representation of __open quantum system dynamics__ with __arbitrary stationary Gaussian baths__ in terms of infinite tensor networks. Specifically, a process tensor representing the system-bath interaction is generated in a uniform matrix product operator form using infinite time evolving block decimation. The process tensor can then be used to compute multi-time dynamics of open quantum systems. 

## Installation

This package is not registered as an official package. To install it directly from github, use

```julia
julia> ]
(myenv) pkg> add https://github.com/uniformTEMPO/UniformTEMPO.jl.git
```

## Getting started

To get started, read the [Introduction](introduction.md#Introduction) page first. We also provide several code examples in the repository under `./examples`. If you need information on a specific funcion, check the [API](reference.md#References) or just use Julia's help framework to directly recover docstrings

```julia
julia> using UniformTEMPO
julia> ?
help?> uniTEMPO
```

## Citation

We are currently working on a citable documentation.

Please cite the papers listed [here](authors.md) if you use this code for your publication.