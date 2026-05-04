# Floquet dynamics of a driven qubit with an Ohmic bath

using UniformTEMPO
using OrdinaryDiffEq

σz = [1 0; 0 -1]
σx = [0 1; 1 0]
ϵ = ω = 1 # driving amplitude and frequency
h_s(t) = σx + ϵ * cos(t * ω) * σx 

T = 2π / ω # driving period
δt = T / 60 # commensurate time-step

ω_c = 1 # cut-off frequency
α = 0.1 # copuling strength
bcf(t) = α * (ω_c / (1 + im * ω_c * t))^2 # bath correlation function

pt = uniTEMPO(σz, δt, bcf, 1e-7);

ρ_0 = [1 0; 0 0] # initial state
n = 1000 # number of time steps
ρ_t = evolve(pt, ρ_0, n, h_s=h_s)

ptf = floquet_process_tensor(pt, h_s, T); # create Floquet PT-MPO

ρ_f = steadystate(ptf) # compute Floquet steady state

x_f = steadystate(ptf, return_full=true) # compute computational Floquet steady state
ρ_t_f = evolve(pt, x_f, n, h_s=h_s) # stationary evolution

using Plots
import LinearAlgebra.tr

t_eval = (0:n) * δt
plot(xlabel="time", ylabel="⟨σx⟩", size=(400, 250), ylims=(-1,0))
plot!(t_eval, [real(tr(ρ * σx)) for ρ in ρ_t], label="quench")
plot!(t_eval, [real(tr(ρ * σx)) for ρ in ρ_t_f], label="stationary")
# savefig("floquet_dynamics.svg")