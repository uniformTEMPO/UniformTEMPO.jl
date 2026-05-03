using ITensors
using UniformTEMPO

# create a nontrivial process tensor
pt = uniTEMPO([-1 0; 0 1], 0.1, t -> exp(-0.5 * abs(t)), 1e-8);

# convert to ITensor representation
pti = to_ITensor(pt, tags=["dephasing"]);

# convert the propagator to Hilbert space rather than Liouville space (optional)
q = pti.q * pti.cmb * pti.cmb'
inds(q)

# initial state for the system
ψ0_ket = ITensor([1, 1] / sqrt(2), inds(pti.cmb, tags="Site+")...);
ψ0_bra = replaceinds(dag(ψ0_ket), inds(ψ0_ket), inds(pti.cmb, tags="Site-"));
ρ0 = ψ0_ket * ψ0_bra;

# set up system+auxiliary initial state
x = ρ0 * pti.v_r;

#evolve a few steps
Nev = 5
for n in 1:Nev
    x = prime(q * x, -1)
end

# compare evolved density matrix against the standard UniformTEMPO evolve
Array(x * pti.v_l, inds(ρ0)) ≈ evolve(pt, Array(ρ0, inds(ρ0)), Nev)[end]