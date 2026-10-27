# quench in a sub-ohmic spin boson model beyond the localization transition with causal TEMPO

using UniformTEMPO
using LinearAlgebra

s = 0.5 # sub-ohmic exponent
α = 0.25 # beyond the localization transition (α_c ≈ 0.1 for Δ / ω_c = 0.1)
ω_c = 10;
Δ = 1;
# J(ω) = 2π α ω_c^(1-s) ω^s exp(-ω/ω_c) at zero temperature, coupling σ_z / 2 (here written for the coupling σ_z)
Γ = sqrt(π) / 2; # Γ(1 + s) for s = 0.5
bcf(t) = 0.5 * α * Γ * ω_c^(1 - s) * (1 / ω_c + im * t)^(-(1 + s));

S = [[1 0]; [0 -1]];
delta_t = 0.05;
N = trunc(Int, 10 / delta_t);
t_ev = (0:N) * delta_t;

# full memory (n_c = N), the bond dimension stays small even at this strong coupling
rhot, bond_dims = causalTEMPO(S, delta_t, bcf, 1e-7, [1 0; 0 0], N; h_s=0.5 * Δ * [0 1; 1 0], return_bond_dims=true);
@show maximum(bond_dims);

using Plots
plot(t_ev, real.([tr(rho * S) for rho in rhot]), lw=2, label="causal TEMPO", xlabel="t", ylabel="⟨σ_z⟩")
