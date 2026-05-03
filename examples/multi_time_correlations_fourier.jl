# multi-time correlations in Fourier space for 2D electronic spectra. Fourier and pathways convention as in  [https://arxiv.org/abs/2603.04970]

using UniformTEMPO
using LinearAlgebra
import SpecialFunctions.polygamma
using Plots

# model
ω_c = 3.04;
ϵ = 3;
α = 0.1;
λ = 2 * α * ω_c;
β = 1 / 7.5;
Ω = 2;
bcf(t) = 2 * α * (polygamma(1, (1 + im * t * ω_c) / β / ω_c) + polygamma(1, (1 - im * t * ω_c + β * ω_c) / β / ω_c)) / (β^2);

v0 = [1; 0; 0];
v1 = [0; 1; 0];
v2 = [0; 0; 1];
H_Sys = -ϵ * v1 * v1' + ϵ * v2 * v2';
S = v2 * v1' + v1 * v2';
ρ0 = v0 * v0';

# Intervention operators (left and right acting)
V = v0 * v2' + v2 * v0' + 0.1 * v0 * v1' + v1 * v0';
Vr = kron(transpose(V), I(3));
Vl = kron(I(3), V);


# compute PT-MPO 
δt = 0.025;
res = 5e-7;
MyPT = uniTEMPO(S, δt, bcf, res);
bond_dim(MyPT)


# stationary linear response function 
ω_eval_linear = LinRange(-10, 20, 500);
L = two_point_correlations_fourier(MyPT, Vl, Vl, ω_eval_linear; h_s=H_Sys);
plot(ω_eval_linear, real.(L), xlabel="ω", ylabel="S")

# 2d spectra frequency grids
ω_eval_τ = LinRange(-10, 10, 200);
ω_eval_t = LinRange(-10, 10, 200);

# three point correlation function (no waiting time) for response function S₂ 
S₂_three_point = three_point_correlations_fourier(MyPT, ρ0, Vr, Vr * Vl, Vl, -ω_eval_τ, ω_eval_t; h_s=H_Sys);
S₂_four_point = four_point_correlations_fourier(MyPT, ρ0, Vr, Vl, Vr, Vl, -ω_eval_τ, 0, ω_eval_t; h_s=H_Sys);
S₂_three_point ≈ S₂_four_point

# Individual pathways contributing to the 2D spectrum
S₁ = four_point_correlations_fourier(MyPT, ρ0, Vl, Vr, Vr, Vl, ω_eval_τ, 0, ω_eval_t; h_s=H_Sys);
S₂ = four_point_correlations_fourier(MyPT, ρ0, Vl, Vr, Vr, Vl, -ω_eval_τ, 0, ω_eval_t; h_s=H_Sys);
S₃ = four_point_correlations_fourier(MyPT, ρ0, Vl, Vr, Vr, Vl, -ω_eval_τ, 0, ω_eval_t; h_s=H_Sys);
S₄ = four_point_correlations_fourier(MyPT, ρ0, Vl, Vr, Vr, Vl, ω_eval_τ, 0, ω_eval_t; h_s=H_Sys);
heatmap(ω_eval_τ, ω_eval_t, real.(S₂))

# An additional waiting time T can be included 
T = 10;
n_T = Int(T / δt);
S₂_T = four_point_correlations_fourier(MyPT, ρ0, Vl, Vr, Vr, Vl, -ω_eval_τ, n_T, ω_eval_t; h_s=H_Sys);
heatmap(ω_eval_τ, ω_eval_t, real.(S₂_T))
