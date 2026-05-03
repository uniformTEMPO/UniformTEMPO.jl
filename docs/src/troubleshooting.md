# Troubleshooting

## `uniTEMPO` convergence

UniTEMPO usually yields plausible dynamics even when simulations are not yet converged with the bond dimension. It is not possible to predict the required accuracy for a given calculation. Hence, convergence with the bond dimension needs to be checked carefully. Convergence with the Trotter step `delta_t` is quadratic and must also be checked. Note that the bond dimension required for convergence is roughly independent of $\delta t$, unlike the tolerance value `tol`. Fixing the tolerance while decreasing the time-step will yield lower bond dimensions and thus convergence with respect to the bond dimension can be lost. When checking for convergence with respect to $\delta t$, make sure you are using process tensors with a similar bond dimension.

## `uniTEMPO` takes forerver to intialize

The most likely issue is that calling the bath correlation function is very expensive. If this is the case, simply precompute the function and provide an interpolation. uniTEMPO integrates over the bath correlation function which requires many function calls.

## I cannot find a value for `n_c` in `uniTEMPO`

First try increasing the tolerance value `tol`. If the tolerance value is around $10^{-14}$ the algorithm will not converge due to floating point accuracy. Note that this limits the maximum achievable bond dimension. To reach larger bond dimensions one has to increase the time step `delta_t`.

This issue can also occur when the the bath correlation function does not properly decay to zero, or it decays algebraically with a small exponent (deeply subohmic). Consider including a low frequency regularization in your spectral density in order to ensure proper decay. 

## Can I use this package for large systems?

The scaling with the dimension of the coupled system Hilbert space is generically quartic which limits the applicability of this method to relatively small systems. The cost for the uniTEMPO contraction can be lowered by activating low rank svd, see [advanced](advanced.md#Notes-on-performance) section.

If the system is large but the subspace coupled to the bath is small, the process tensor can be computed separately for the coupled subspace. However, the evolution of the full system must then be implemented by hand. We recommend exporting the process tensor to ITensors and then performing the full system evolution within ITensors.

## Strongly peaked spectral densities

Peaked spectral densities can be more challenging to converge with uniTEMPO. The uniTEMPO algorithm requires a decaying bath correlation function. Peaked spectral densities lead to a slow bath correlation function decay which in turn will lead to a large memory cutoff `n_c` and often large bond dimensions.

## Convergence guidelines for driven system

For a periodic driving it is recommended to choose a time step which is commensurate with the period of the time-periodic Hamiltonian. For driven systems it is moreover recommended to load `OrdinaryDiffEq` package in order to activate exact integration of the local dynmaics. Otherwise the Trotter error will be linear.

## `steadystate` result does not match `evolve`

For systems with symmetries, the dynamics has have several distinct steady states, which are realized by choosing an initial state in the particular sector. On the other hand, [`steadystate`](@ref steadystate) function returns only the eigenstate with the smalles (closest to zero) decay rate. In the situation where multiple steady states exist, it will return a random combination of all zero-eigenvectors. 

If the steadystate is not a physical state, it could be that the spectral gap is small such that the Krylov solver does not converge. In this case extracting the steadystate from full ED via the keyword argument `ED = true` could be more stable. In this case it can be instructive to analyze the spectrum of the PT-MPO via [`spectrum`](@ref spectrum).