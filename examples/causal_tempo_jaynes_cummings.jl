# causal TEMPO vs uniTEMPO and the exact pseudomode solution for the damped Jaynes-Cummings model
# (non-commuting couplings σx, σy), Fig. 1 of [https://arxiv.org/abs/2603.23432].

using UniformTEMPO
using LinearAlgebra

γ = 2
Ω = 2
g = 1
n_bar = 0.25

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]
sm = 0.5 * (sx - im * sy)

bcf0(t) = g^2 * (1 + n_bar) * exp(-γ * abs.(t) - Ω * im * t) + g^2 * n_bar * exp(-γ * abs.(t) + Ω * im * t)
bcf1(t) = g^2 * (1 + n_bar) * exp(-γ * abs.(t) - Ω * im * t) - g^2 * n_bar * exp(-γ * abs.(t) + Ω * im * t)
bcfv(t) = [bcf0(t) -im*bcf1(t); im*bcf1(t) bcf0(t)] / 4

Hsys = 0.5 * sx
ψ0s = [1.0; 1.0] / sqrt(2)
ρ0 = ψ0s * ψ0s'
delta_t = 0.1
n = trunc(Int, 5 / delta_t)

### exact solution via pseudomode ###
Nb = 25
function bdestroy(dim::Int)
    a = zeros(ComplexF64, dim, dim)
    for k in 1:(dim-1)
        a[k, k+1] = sqrt(k)
    end
    return a
end
a = bdestroy(Nb)
adag = a'
H = g * (kron(sm, adag) + kron(sm', a)) + Ω * kron(I(2), adag * a)
L1 = sqrt(2 * γ * (1 + n_bar)) * kron(I(2), a)
L2 = sqrt(2 * γ * n_bar) * kron(I(2), adag)
pt0_pm = UniformTEMPO.UniformPTMPO_from_GKSL(delta_t, Ω * adag * a, [sqrt(2 * γ * (1 + n_bar)) * a, sqrt(2 * γ * n_bar) * adag], ones(1, 1))
pt_pm = UniformTEMPO.UniformPTMPO_from_GKSL(delta_t, H, [L1, L2], steadystate(pt0_pm))
rt_pm = evolve(pt_pm, ρ0, n; h_s=Hsys)

### uniTEMPO ###
pt = uniTEMPO([sx, sy], delta_t, bcfv, 1e-9)
rt_u = evolve(pt, ρ0, n; h_s=Hsys)

### causal TEMPO ###
rt_c, bond_dims = causalTEMPO([sx, sy], delta_t, bcfv, 1e-8, ρ0, n; h_s=Hsys, return_bond_dims=true)

err(r1, r2) = maximum(norm(r1[i] - r2[i]) for i in eachindex(r1))
println("uniTEMPO:     bond dimension $(bond_dim(pt)), max error vs pseudomode $(err(rt_u, rt_pm))")
println("causal TEMPO: bond dimension $(maximum(bond_dims)), max error vs pseudomode $(err(rt_c, rt_pm))")

using Plots
t_ev = (0:n) * delta_t
plot(t_ev, real.([tr(sz * r) for r in rt_pm]), lw=2, label="pseudomode", xlabel="t", ylabel="⟨σ_z⟩")
plot!(t_ev, real.([tr(sz * r) for r in rt_u]), ls=:dash, label="uniTEMPO")
plot!(t_ev, real.([tr(sz * r) for r in rt_c]), ls=:dot, lw=2, label="causal TEMPO")
