using UniformTEMPO
using Test

@testset "uniform_process_tensors" begin
    include("test_uniform_process_tensors.jl")
end

@testset "uniform_mps" begin
    include("test_uniform_MPS.jl")
end

@testset "unitempo_base" begin
    include("test_unitempo.jl")
end

@testset "unitempo_superfilter" begin
    include("test_unitempo_superfilter.jl")
end

@testset "causal_tempo" begin
    include("test_causal_tempo.jl")
end

@testset "pseudomode" begin
    include("test_pseudomode.jl")
end
