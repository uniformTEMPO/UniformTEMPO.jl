using UniformTEMPO
using Test
using LinearAlgebra
using Random
Random.seed!(1234)

# check trivial PT-MPO
s_dim = 2
delta_t = 0.1
pt = UniformPTMPO(s_dim, delta_t)
@test pt(rand(1:4, 100)) ≈ 1.0

# nontrivial PT-MPO (gaussian pseudomode)
bcf(t) = exp(-abs(t) - im * t)
# create Hamiltonian and Lindblad operator
function bdestroy(dim::Int)
    a = zeros(ComplexF64, dim, dim)
    for k in 1:(dim-1)
        a[k, k+1] = sqrt(k)
    end
    return a
end
a = bdestroy(5)
adag = a'
H = kron([1 0; 0 -1], adag + a) + kron(I(2), adag * a)
L = sqrt(2) * kron(I(2), a)
ρ0b = zeros(5, 5)
ρ0b[1, 1] = 1.0

pt = UniformTEMPO.UniformPTMPO_from_GKSL(delta_t, H, [L], ρ0b);
path = rand(1:4, 100);
val_ex = UniformTEMPO.exact_gaussian_influence([1 0; 0 -1], delta_t, bcf, path);
@test isapprox(val_ex, pt(path); atol=1e-4)

# check iMPS truncation
pt_trunc = UniformTEMPO.truncate(pt, 1e-6);
@test isapprox(pt_trunc(path), pt(path); atol=1e-4);

# check channel
chan = UniformTEMPO.choi_channel(pt, 100, h_s = [0 1; 1 0])[end]
@test isapprox(tr(chan), 1.0; atol=1e-6) # check trace preservation
@test all(real.(eigvals(chan)) .>= -1e-8) # check positivity

# check evolve
ρ0 = rand(ComplexF64, 2, 2)
ρt = UniformTEMPO.evolve(pt, ρ0, 100; h_s = [0 1; 1 0])
ρt2 = UniformTEMPO.evolve(pt, ρ0, 100; h_s = t->[0 1; 1 0])
@test ρt2 ≈ ρt

# test channel
Φt = UniformTEMPO.channel(pt, 100; h_s = [0 1; 1 0])
Φt2 = UniformTEMPO.channel(pt, 100; h_s = t->[0 1; 1 0])
@test Φt2 ≈ Φt
@test ρt ≈ [reshape(Φ * ρ0[:], 2, 2) for Φ in Φt]