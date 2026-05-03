# this script computes the dynamics in Fig. 4 of [https://arxiv.org/abs/2603.23432]. 

using SpecialFunctions
using Plots
using LinearAlgebra
using UniformTEMPO

"""
    square_lattice_bcf(t::Real, α::Real, w0::Real, J::Real, d::Int; Δx::Real=0)

BCF for a square lattice bath in d dimensions with nearest neighbor coupling J and onsite energy w0.
"""
function square_lattice_bcf(t::Real, α::Real, w0::Real, J::Real, d::Int; Δx::Real=0)
    return α * exp(-im * w0 * t) * (SpecialFunctions.besselj0(2J * t))^(d - 1) * SpecialFunctions.besselj(Δx, 2J * t) * (im * 1.0)^(Δx)
end

α = 0.1
d = 3 # space dimension
J = 1
w0 = 2 * J * d # we must have w0 >= 2*d*J for a stable crystal
Δx = 1

delta_t = 0.05

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]

Δ = 0.0 # detuning

### no RWA calculation ###

h_s_lab = kron(sz, I(2)) / 2 * (w0 + Δ) + kron(I(2), sz) / 2 * (w0 + Δ) # lab frame Hamiltonian

bcfv(t) = [square_lattice_bcf(t, α, w0, J, d) square_lattice_bcf(t, α, w0, J, d; Δx=Δx); square_lattice_bcf(t, α, w0, J, d; Δx=-Δx) square_lattice_bcf(t, α, w0, J, d)]

pt = uniTEMPO([kron(sx, I(2)), kron(I(2), sx)], delta_t, bcfv, 1e-7; low_rank_svd=true, truncation=:abs, auto_nc=false, n_c=2048); # auto_nc struggles with this oscillating bcf, its better to fix a cutoff manually.

steadystate(pt; h_s=h_s_lab)

### RWA calculation ###

h_s_rot = kron(sz, I(2)) / 2 * Δ + kron(I(2), sz) / 2 * Δ # rotating frame Hamiltonian

function bcfv_RWA(t)
    return [1 -im 0 0; im 1 0 0; 0 0 1 -im; 0 0 im 1] / 4 * square_lattice_bcf(t, α, 0, J, d) + [0 0 1 -im; 0 0 im 1; 0 0 0 0; 0 0 0 0] / 4 * square_lattice_bcf(t, α, 0, J, d; Δx=Δx) + [0 0 0 0; 0 0 0 0; 1 -im 0 0; im 1 0 0] / 4 * square_lattice_bcf(t, α, 0, J, d; Δx=-Δx)
end

sops = [kron(sx, I(2)), kron(sy, I(2)), kron(I(2), sx), kron(I(2), sy)]
@time pt_RWA = uniTEMPO(sops, delta_t, bcfv_RWA, 1e-7, low_rank_svd=true, truncation=:abs, auto_nc=false, n_c=1024);

bond_dim(pt_RWA)

steadystate(pt_RWA)

Ω = 0.1; # drive strength
h_s_lab_drive(t) = h_s_lab + Ω * kron(sx, I(2)) * cos(w0 * t);
h_s_rot_drive = h_s_rot + Ω / 2 * kron(sx, I(2));

n_steps = 3000
t_eval = (0:n_steps) * delta_t

n_a = kron([1 0; 0 0], I(2)) # operator to measure occupation of emitter a
n_b = kron(I(2), [1 0; 0 0]) # operator to measure occupation of emitter b
r0 = [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0] # initial state with emitter a excited

# no RWA dynamics (lab frame)
rt = evolve(pt, r0, n_steps; h_s=h_s_lab_drive)
# RWA dynamics (rotating frame)
rt_rwa = evolve(pt_RWA, r0, n_steps; h_s=h_s_rot_drive)

plot(xlabel="Jt", ylabel="⟨n⟩")
plot!(t_eval, real.(UniformTEMPO.expect(n_a, rt)), label="emitter a")
plot!(t_eval, real.(UniformTEMPO.expect(n_a, rt_rwa)), label="emitter a RWA")

plot(xlabel="Jt", ylabel="⟨n⟩")
plot!(t_eval, real.(UniformTEMPO.expect(n_b, rt)), label="emitter b")
plot!(t_eval, real.(UniformTEMPO.expect(n_b, rt_rwa)), label="emitter b RWA")