using UniformTEMPO
using LinearAlgebra
using Test

# Damped Jaynes-Cummings model at finite temperature (thermal pseudomode, Fig. 1 of arXiv:2603.23432): a spin coupled
# through σx and σy to one lossy bosonic mode, equivalent to a Gaussian bath with the two-channel correlation matrix
# bcfv. The reference is the Lindblad evolution of spin + mode (UniformPTMPO from the GKSL generator, mode thermal).

sx = ComplexF64[0 1; 1 0]
sy = ComplexF64[0 -im; im 0]
sm = (sx - im * sy) / 2

γ, Ω, g, n_bar = 2.0, 2.0, 1.0, 0.25
bcf0(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) + g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
bcf1(t) = g^2 * (1 + n_bar) * exp(-γ * abs(t) - Ω * im * t) - g^2 * n_bar * exp(-γ * abs(t) + Ω * im * t)
bcfv(t) = [bcf0(t) -im*bcf1(t); im*bcf1(t) bcf0(t)] / 4
h_s = 0.5 * sx
ψ0 = ComplexF64[1; 1] / sqrt(2)
ρ0 = ψ0 * ψ0'
dt, n = 0.05, 100

Nb = 20
a = ComplexF64.(diagm(1 => sqrt.(1.0:Nb-1)))
H = g * (kron(sm, a') + kron(sm', a)) + Ω * kron(I(2), a' * a)
L = [sqrt(2γ * (1 + n_bar)) * kron(I(2), a), sqrt(2γ * n_bar) * kron(I(2), a')]
pt_mode = UniformTEMPO.UniformPTMPO_from_GKSL(dt, Ω * a' * a, [sqrt(2γ * (1 + n_bar)) * a, sqrt(2γ * n_bar) * a'], ones(1, 1))
ρ_ref = evolve(UniformTEMPO.UniformPTMPO_from_GKSL(dt, H, L, steadystate(pt_mode)), ρ0, n; h_s=h_s)
err(ρt) = maximum(norm(ρt[i] - ρ_ref[i]) for i in eachindex(ρt))

@testset "thermal pseudomode: uniTEMPO superfilter and dense conversion" begin
    pt = uniTEMPO_superfilter([sx, sy], dt, bcfv, 1e-9)
    ρ_sparse = evolve(pt, ρ0, n; h_s=h_s)
    @test err(ρ_sparse) < 1e-2 # Trotter error of the time step dt = 0.05
    ρ_dense = evolve(UniformPTMPO(pt), ρ0, n; h_s=h_s)
    @test maximum(norm(ρ_sparse[i] - ρ_dense[i]) for i in 1:n+1) < 1e-10
end

@testset "thermal pseudomode: causal TEMPO superfilter" begin
    ρ_c = causalTEMPO_superfilter([sx, sy], dt, bcfv, 1e-9, ρ0, 40; h_s=h_s) # t = 2 (the cost grows with the number of memory lines)
    @test err(ρ_c) < 1e-2
end
