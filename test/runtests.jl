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