# Confirm the Shiba relation for spin boson

using UniformTEMPO
import SpecialFunctions.gamma

s = 0.5;
ω_c = 5;
α = 0.1;

J(ω) = α * (ω)^s * exp(-ω / ω_c)
bcf(t) = α / pi * gamma(s + 1) * (ω_c / (1 + im * ω_c * t))^(s + 1);

S = [[0 1]; [1 0]];
Δ = 0.1;
MyPT = uniTEMPO(S, Δ, bcf, 1e-9);
@show bond_dim(MyPT);

H_sys = [[1 0]; [0 -1]]

ω_eval = LinRange(0, 0.1, 1000)

χ = susceptibility(MyPT, S, S, ω_eval; h_s = H_sys)

# susceptibility function is short for this two-point correlation.
# χ = im * (two_point_correlations_fourier(MyPT, kron(I(2),S) - kron(transpose(S),I(2)), kron(I(2),S), ω_eval; h_s = H_sys)) 

χ0 = χ[argmin(abs.(ω_eval))]

# We find the expected agreement at low frequencies, indicating algebraic decay of the correlation function.
using Plots
plot(ω_eval, @. sign(ω_eval) * real(χ0)^2 * J(abs.(ω_eval)); label="Shiba Relation", ls=:dash, xlabel="ω", ylabel="χ''", xlims=(-0.01, 0.1))
plot!(ω_eval, imag.(χ), label="uniTEMPO")