module UniformTEMPO
using LinearAlgebra
using TensorOperations
using OMEinsum
using ProgressMeter
import LowRankApprox.psvd
import Cubature.hcubature_v
import KrylovKit.eigsolve

abstract type LocalChannelBackend end
struct LocalChannelBasic <: LocalChannelBackend end
struct LocalChannelODE <: LocalChannelBackend end
const local_channel_default_backend = Ref{LocalChannelBackend}(LocalChannelBasic())

abstract type EigsolveBackend end
struct EigsolveKrylovKit <: EigsolveBackend end
struct EigsolveFullEDBackend <: EigsolveBackend end
struct EigsolveGeneric <: EigsolveBackend end
const eigsolve_default_backend = Ref{EigsolveBackend}(EigsolveFullEDBackend())

export UniformPTMPO, *, +, uniTEMPO, evolve, channel, choi_channel, steadystate, two_point_correlations, process_tensor, spectrum, susceptibility, two_point_correlations_fourier, three_point_correlations_fourier, four_point_correlations_fourier, bond_dim, floquet_process_tensor, truncate, to_ITensor, pt_to_choi, include_system_hamiltonian

include("uniform_process_tensors.jl")
include("unitempo_base.jl")

end