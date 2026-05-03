# test uniTEMPO on spin boson with psuedomode bath (Lorentzian spectral density)

using UniformTEMPO
using LinearAlgebra

γ = 1
Ω = 1
g = 0.5
bcf(t) = g^2 * exp(-γ * abs.(t) - Ω * im * t)

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]

### uniTEMPO ###
s_op = sx
MyPT = uniTEMPO(s_op, 0.05, bcf, 1e-10);

### exact solution via pseudomode ###
Nb = 5 # number of pseudomode levels
function bdestroy(dim::Int)
    a = zeros(ComplexF64, dim, dim)
    for k in 1:(dim-1)
        a[k, k+1] = sqrt(k)
    end
    return a
end
a = bdestroy(Nb)
adag = a'
H = g * kron(s_op, a+adag) + Ω * kron(I(2), adag * a)
L = sqrt(2 * γ) * kron(I(2), a)
ρ0b = zeros(Nb, Nb)
ρ0b[1, 1] = 1.0
MyPT_pm = UniformTEMPO.UniformPTMPO_from_GKSL(MyPT.delta_t, H, [L], ρ0b);

### evolve and compare ###
Hsys = 0.2 * sz + 0.4 * sy
ψ0s = [1.0; 1.0] / sqrt(2)
t = 40
rt = evolve(MyPT, ψ0s * ψ0s', trunc(Int, t / MyPT.delta_t); h_s=Hsys)
rt_pm = evolve(MyPT_pm, ψ0s * ψ0s', trunc(Int, t / MyPT.delta_t); h_s=Hsys)

using Plots
plot(real.(UniformTEMPO.expect(sx, rt)))
plot!(real.(UniformTEMPO.expect(sx, rt_pm)))
plot!(real.(UniformTEMPO.expect(sy, rt)))
plot!(real.(UniformTEMPO.expect(sy, rt_pm)))
plot!(real.(UniformTEMPO.expect(sz, rt)))
plot!(real.(UniformTEMPO.expect(sz, rt_pm)))