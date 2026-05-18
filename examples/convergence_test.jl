using UniformTEMPO
import SpecialFunctions.zeta, SpecialFunctions.gamma
using LinearAlgebra
using Plots

include("../utils/convergence.jl");


s = 1; 
ω_c = 5;
α = 0.2;

β = 1;
bcf_0(t) = α / pi * gamma(s + 1) * (ω_c / (1 + im * ω_c * t))^(s + 1);
bcf_th(t) = 2 * α / (pi * β^(s + 1)) * gamma(s + 1) * real(zeta(s + 1, (1 + β * ω_c + im * ω_c * t) / (β * ω_c)));
bcf(t) = bcf_0(t) + bcf_th(t);

S = [[0 1]; [1 0]];
H_sys = [[1 0]; [0 -1]];

# convergence grid: svd accuracy and trotter time step 
accuracy = 10 .^ LinRange(-5, -12, 4);
trotter_steps = reverse(LinRange(0.01, 0.1, 4));

# Testing convergence routine
# Argument structure is the same as computing the process tensor 
# max_rank is an optional argument, but recommended 
bdim, values, index = convergence(S, trotter_steps, bcf, accuracy; max_rank = 250) do MyPT 
    
    ω_eval= LinRange(0, 10, 2000)
    susceptibility(MyPT, S, S, ω_eval; h_s = H_sys)

end;


# The bond dimension can be visualized
heatmap(something.(bdim, NaN), cmap = :heat, xlabel = "Accuracy", ylabel = "Trotter time step")
χ, mcχ = maximum_common_χ(bdim, index; ref = 250);
contour!(something.(bdim, NaN), c=:black, levels = [χ])


# Convergence: bond dimension χ
t = 8; trotter_steps[t]
δε = Array{Float64}(undef, index[t]-1);
δχ = similar(δε);

for k in 1:(index[t]-1)
    δε[k] = sum(abs.(values[t, k+1] .- values[t, k]))/length(values[t, k])
    δχ[k] = abs(bdim[t, k+1] - bdim[t, k])
end
scatter(bdim[t, 1:(index[t]-1)], δε ./ δχ, lw = 3, yscale = :log10, xscale = :log10)

# Convergence: trotter step Δ
δϵ = Array{Float64}(undef, length(trotter_steps)-1);
δΔ = similar(δϵ);
χ, mcχ = maximum_common_χ(bdim, index; ref = 250, tol = 10);
for t in eachindex(δϵ)
    δϵ[t] = sum(abs.(values[t,mcχ[t]] .- values[t+1,mcχ[t+1]]))/length(values[t, mcχ[t]])
    δΔ = abs(trotter_steps[t+1]-trotter_steps[t])
end

plot(trotter_steps[begin:end-1], δϵ ./ δΔ)








# Bond dimension, fixed trotter step 
t = 10
colors = cgrad(:blues, index[t]-1, categorical=true);

plt = plot(xlabel = "ω", ylabel = "diff")
for k in 1:index[t]-1 
    diff = values[t, index[t]] .- values[t, k]
    plot!(plt, real.(diff), lw = 2.5, lc = colors[k], label = bdim[t, k])
end
plot!(plt)


# Trotter step, comparing last computation
plt = plot(xlabel = "ω", ylabel = "value")
for t in eachindex(values[:, 1])
    plot!(plt, imag.(values[t, index[t]-1]))
end
plot!(plt)




#plotting the spectrum 
MyPT = uniTEMPO(S, 0.05, bcf, 1e-8);
bond_dim(MyPT)

plotly();
λ, v = spectrum(MyPT; h_s = H_sys);
scatter(λ)
