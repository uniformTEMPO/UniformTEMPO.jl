# usage of the UniformTEMPO built-in convergence routines

using UniformTEMPO
import SpecialFunctions.zeta, SpecialFunctions.gamma
using Plots

include("../utils/convergence.jl");

# model as in `examples/spin_boson_fdt.jl`
begin
    s = 1; ω_c = 5; α = 0.2; β = 1;

    bcf_0(t) = α / pi * gamma(s + 1) * (ω_c / (1 + im * ω_c * t))^(s + 1);
    bcf_th(t) = 2 * α / (pi * β^(s + 1)) * gamma(s + 1) * real(zeta(s + 1, (1 + β * ω_c + im * ω_c * t) / (β * ω_c)));
    bcf(t) = bcf_0(t) + bcf_th(t);

    S = [[0 1]; [1 0]];
    H_sys = [[1 0]; [0 -1]];
end;
# FDT demands that Sₛ = 2 * (n_B.(ω_eval) .+ 1) .* imag.(χ) for a thermal bath,
# with Sₛ the power spectral density, χ the susceptibility, and n_B the bose distribution 
# We (convergence) check that the identity is satisfied in the frequency domain. 



# convergence grid: svd accuracy and trotter time step 
accuracy = 10 .^ LinRange(-5, -12, 20);
trotter_steps = reverse(LinRange(0.01, 0.1, 20));

# A label for the convergence test can be specified
label = "fluctuation dissipation theorem";

# A <path> and <filename> for the convergence run can be specified. Each unique convergence run creates a folder <path>/<filename>, a convergece file <path>/<filename>/<filename>.jld2
path = pwd();
filename = "fdt_spin_boson_α_$(α)_β_$(β)"; # with or w/out ".jld2" suffix

# Additional metadata can be saved in dictionary form 
metadata = Dict{String,Any}(
    "computed_value"  => "Sₛ - 2 * (n_B(ω_eval) + 1) * imag(χ)",
    "omega_range" => (-5, 5),
    "n_omega"     => 1000,
    "H_sys"       => string(H_sys));

# The computed process tensor can be saved for each (trotter_step, accuracy) combination
# False by default. If true, an additional folder is created at <path>/<filename>/pt_<filename>
pt_save = true;

# A maximum rank can be specified. This avoid running into memory issues whenever the bond dimension becomes very large for the (trotter_step, accuracy) combination. This option is *not* `cap_rank`.
max_rank = 350;


# The convergence function is called with a `do` block. 
# It returns a 2d array of bond dimensions `bdim`, of `values`, and 1d array of the maximum `index` (of accuracy) reached for every trotter step before `max rank` was reached.

# The arguments follow the same ordering in the `uniTEMPO(...) function`. 
# Any additional uniTEMPO argument *must* be passed after the convergence specific arguments
bdim, values, index = convergence(  S, trotter_steps, bcf, accuracy;
                                    label, metadata, pt_save, filename, path, # convergence arguments
                                    max_rank # uniTEMPO argument
                                ) do MyPT
    # quantity to evaluate     
    ω_eval = LinRange(-5, 5, 1000)
    χ = susceptibility(MyPT, S, S, -ω_eval; h_s = H_sys) 
    Sₛ = 2 * real.(two_point_correlations_fourier(MyPT, S, S, -ω_eval; h_s = H_sys)) 
    n_B(w) = 1 / (exp(β * w) - 1) 

    # return fdt value
    2 * (n_B.(ω_eval) .+ 1) .* imag.(χ) - Sₛ
end;

# In case the convergence run is terminated before its completion, it can be resumed from the checkpoint file
bdim, values, index = resume_from_checkpoint(S, bcf; path, filename, pt_save = true) do MyPT
    # quantity to evaluate     
    ω_eval = LinRange(-5, 5, 1000)
    χ = susceptibility(MyPT, S, S, -ω_eval; h_s = H_sys) 
    Sₛ = 2 * real.(two_point_correlations_fourier(MyPT, S, S, -ω_eval; h_s = H_sys)) 
    n_B(w) = 1 / (exp(β * w) - 1) 

    # return fdt value
    2 * (n_B.(ω_eval) .+ 1) .* imag.(χ) - Sₛ
end







