# quench in an ohmic spin boson model at strong coupling

using UniformTEMPO

ω_c = 10;
α = 0.7; # this is very strong coupling
bcf(t) = 0.5 * α * (ω_c / (1 + im * ω_c * t))^2;

S = [[1 0]; [0 -1]];
delta_t = 0.01;
MyPT = uniTEMPO(S, delta_t, bcf, 1e-9);
@show bond_dim(MyPT);

N = trunc(Int, 20 / delta_t);
t_ev = (0:N) * delta_t;
rhot = evolve(MyPT, [1 0; 0 0], N, h_s=[0 1; 1 0]);

using Plots
plot(t_ev, real.([tr(rho * [1 0; 0 -1]) for rho in rhot]), lw=2, label=bond_dim(MyPT))