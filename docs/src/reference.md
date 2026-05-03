# References

```@docs
UniformPTMPO
Base.:+(::UniformPTMPO, ::UniformPTMPO)
Base.:*(::UniformPTMPO, ::AbstractMatrix{<:Number})
Base.:*(::AbstractMatrix{<:Number}, ::UniformPTMPO)
Base.:^(::UniformPTMPO, ::Real)
include_system_hamiltonian
bond_dim
uniTEMPO 
evolve
channel
choi_channel
steadystate
spectrum
two_point_correlations
two_point_correlations_fourier
three_point_correlations_fourier
four_point_correlations_fourier
susceptibility
process_tensor
pt_to_choi
floquet_process_tensor
UniformTEMPO.truncate
to_ITensor
```