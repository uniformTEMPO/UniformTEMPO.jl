# test non-additive uniTEMPO on Jaynes-Cummings model, as in Fig. 1 of [https://arxiv.org/abs/2603.23432].

using UniformTEMPO
using LinearAlgebra

γ = 2
Ω = 2
g = 2
bcf(t) = g^2 * exp(-γ * abs.(t) - Ω * im * t)
n_bar = 0.25

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]
sm = 0.5 * (sx - im * sy)

### uniTEMPO ###
s1 = sx
s2 = sy
bcf0(t) = g^2 * (1+n_bar) * exp(-γ * abs.(t) - Ω * im * t) + g^2 * n_bar * exp(-γ * abs.(t) + Ω * im * t)
bcf1(t) = g^2 * (1+n_bar) * exp(-γ * abs.(t) - Ω * im * t) - g^2 * n_bar * exp(-γ * abs.(t) + Ω * im * t)
bcfv(t) = [bcf0(t) -im*bcf1(t); im*bcf1(t) bcf0(t)] / 4
MyPT = uniTEMPO([s1, s2], 0.05, bcfv, 1e-9);

### exact solution via pseudomode ###
Nb = 25 # number of pseudomode levels
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
L1 = sqrt(2 * γ * (1+n_bar)) * kron(I(2), a)
L2 = sqrt(2 * γ * n_bar) * kron(I(2), adag)
# to get the steady state of uncoupled pseudomode (thermal ic)
MyPT0_pm = UniformTEMPO.UniformPTMPO_from_GKSL(MyPT.delta_t, Ω*adag*a,[sqrt(2 * γ * (1+n_bar)) * a, sqrt(2 * γ * n_bar) * adag], ones(1,1));
steadystate(MyPT0_pm)
MyPT_pm = UniformTEMPO.UniformPTMPO_from_GKSL(MyPT.delta_t, H, [L1, L2], steadystate(MyPT0_pm));

### evolve and compare ###
Hsys = 0.5 * sx
ψ0s = [1.0; 1.0] / sqrt(2)
t = 10
rt = evolve(MyPT, ψ0s * ψ0s', trunc(Int, t / MyPT.delta_t); h_s=Hsys)
rt_pm = evolve(MyPT_pm, ψ0s * ψ0s', trunc(Int, t / MyPT.delta_t); h_s=Hsys)
t_eval = (0:size(rt, 1)-1) * MyPT.delta_t

using Plots

scatterres=10
plot(t_eval, real.(UniformTEMPO.expect(sx, rt)), size=(400, 250), label="uniTEMPO x", legend=:bottomright, xlabel="time Ωt", ylabel="⟨σ_i⟩", ylims=(-1,1))
plot!(t_eval, real.(UniformTEMPO.expect(sy, rt)), label="uniTEMPO y")
plot!(t_eval, real.(UniformTEMPO.expect(sz, rt)), label="uniTEMPO z")
scatter!(t_eval[1:scatterres:end], real.(UniformTEMPO.expect(sx, rt_pm))[1:scatterres:end],marker=:x, label="reference x")
scatter!(t_eval[1:scatterres:end], real.(UniformTEMPO.expect(sy, rt_pm))[1:scatterres:end],marker=:star5, label="reference y")
scatter!(t_eval[1:scatterres:end], real.(UniformTEMPO.expect(sz, rt_pm))[1:scatterres:end],marker=:+, label="reference z")