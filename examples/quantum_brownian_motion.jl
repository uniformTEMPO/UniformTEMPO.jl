# quantum brownian motion example from [https://arxiv.org/abs/2603.06840]

using UniformTEMPO
using LinearAlgebra
using SpecialFunctions

# bcf from their QED example without Purcell filter (strength p = 0)
# if p != 0, the bcf needs to be evaluated numerically
ghz = 1;
ωr = 7.5 * ghz;
ω_c = 3 * ghz;
s = 1;
η = 0.002;
ωq = 5.3 * ghz;
g = 0.211 * ghz;
p = 0;
Hp(ω) = 1 - p * exp(-150 * ω^2);
J(ω) = 2 * η * ω * exp(-ω / ω_c) * Hp(ω - ωq);
bcf(t) = 2 * η * gamma(s + 1) * (ω_c / (1 + im * ω_c * t))^(s + 1); # bcf for p = 0
Nb = 20 # number of HO levels

function bdestroy(dim::Int)
    a = zeros(ComplexF64, dim, dim)
    for k in 1:(dim-1)
        a[k, k+1] = sqrt(k)
    end
    return a
end
a = bdestroy(Nb)
S = a + a'

delta_t = 0.1
S = diagm(eigvals(S)) # choose diagonal basis (convenient for accuracy check below)

@time f1 = uniTEMPO(S, delta_t, bcf, 1e-3; low_rank_svd=false, truncation=:abs); # using full svd is very slow for large Nb
@time f2 = uniTEMPO(S, delta_t, bcf, 8e-8; low_rank_svd=true, truncation=:abs); # low rank svd
@time f3 = uniTEMPO(S, delta_t, bcf, 1e-7; low_rank_svd=false, truncation=:abs, svd_filtering_tol=1e-8); # svd filtering from [https://arxiv.org/abs/2603.06840]

bond_dim(f1), bond_dim(f2), bond_dim(f3)

# check compression accuracies
path = rand(1:Nb^2, 100)
exact = UniformTEMPO.exact_gaussian_influence(S, delta_t, bcf, path)
abs(f1(path) - exact), abs(f2(path) - exact), abs(f3(path) - exact)