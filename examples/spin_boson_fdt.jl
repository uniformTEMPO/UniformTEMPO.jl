# confirm fluctuation dissipation relation for spin boson

using UniformTEMPO
import SpecialFunctions.zeta, SpecialFunctions.gamma

s = 1; # s=1 is Ohmic
ω_c = 5;
α = 0.2;

β = 1;
bcf_0(t) = α / pi * gamma(s + 1) * (ω_c / (1 + im * ω_c * t))^(s + 1);
bcf_th(t) = 2 * α / (pi * β^(s + 1)) * gamma(s + 1) * real(zeta(s + 1, (1 + β * ω_c + im * ω_c * t) / (β * ω_c)));
bcf(t) = bcf_0(t) + bcf_th(t);

S = [[0 1]; [1 0]];
Δ = 0.1;
MyPT = uniTEMPO(S, Δ, bcf, 1e-9);
@show bond_dim(MyPT);

H_sys = [[1 0]; [0 -1]]

ω_eval = LinRange(-5, 5, 1000)

χ = susceptibility(MyPT, S, S, -ω_eval; h_s = H_sys) # susceptibility (check that sign is correct, as it has been updated in the core function)
Sₛ = 2 * real.(two_point_correlations_fourier(MyPT, S, S, -ω_eval; h_s = H_sys)) # power spectral density
n_B(w) = 1 / (exp(β * w) - 1) # bose distribution

# FDT demands that Sₛ = 2 * (n_B.(ω_eval) .+ 1) .* imag.(χ) for a thermal bath
using Plots
plot(ω_eval, 2 * (n_B.(ω_eval) .+ 1) .* imag.(χ), lw=3, label="2(1+n)Imχ", xlabel="ω")
plot!(ω_eval, Sₛ, lw=1.5, ls=:dash, label="Sₛ")