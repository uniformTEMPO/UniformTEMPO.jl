using UniformTEMPO
using LinearAlgebra
using Test

sx = ComplexF64[0 1; 1 0]
sy = ComplexF64[0 -im; im 0]
sz = ComplexF64[1 0; 0 -1]

@testset "uniTEMPO superfilter: spin boson agrees with uniTEMPO" begin
    # for one spin the quadratic bases span all labels; the second-order gates leave an O(dt^6) error per gate
    bcf(t) = 0.5 * exp(-abs(t) - im * t)
    ρ0 = ComplexF64[1 0; 0 0]
    ρu = evolve(uniTEMPO(sz, 0.1, bcf, 1e-10), ρ0, 100; h_s=sx)
    pt = uniTEMPO_superfilter(sz, 0.1, bcf, 1e-10)
    ρs = evolve(pt, ρ0, 100; h_s=sx)
    @test pt isa SparseUniformPTMPO
    @test maximum(norm(ρu[i] - ρs[i]) for i in 1:101) < 1e-4
    @test maximum(abs(tr(ρ) - 1) for ρ in ρs) < 1e-6 # truncation level, as uniTEMPO (1.9e-7)
end

@testset "uniTEMPO superfilter: non-commuting couplings agree with uniTEMPO" begin
    # damped Jaynes-Cummings model (two non-commuting couplings, two bath channels), as in test_causal_tempo.jl
    γ, Ω, g, n_bar = 2, 2, 1, 0.25
    bcf0(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) + g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
    bcf1(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) - g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
    bcfv(t) = [bcf0(t) -im*bcf1(t); im*bcf1(t) bcf0(t)] / 4
    ρ0 = ComplexF64[0.5 0.5; 0.5 0.5]
    ρu = evolve(uniTEMPO([sx, sy], 0.1, bcfv, 1e-11), ρ0, 30; h_s=0.5 * sx)
    ρs = evolve(uniTEMPO_superfilter([sx, sy], 0.1, bcfv, 1e-11), ρ0, 30; h_s=0.5 * sx)
    @test maximum(norm(ρu[i] - ρs[i]) for i in 1:31) < 1e-3
    # time-dependent Hamiltonian
    ht(t) = 0.5 * cos(t) * sx
    ρu = evolve(uniTEMPO([sx, sy], 0.1, bcfv, 1e-11), ρ0, 30; h_s=ht)
    ρs = evolve(uniTEMPO_superfilter([sx, sy], 0.1, bcfv, 1e-11), ρ0, 30; h_s=ht)
    @test maximum(norm(ρu[i] - ρs[i]) for i in 1:31) < 1e-3
end
