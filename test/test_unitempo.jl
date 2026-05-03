using UniformTEMPO
using Test
using Random
Random.seed!(1234)

bcf(t) = exp(-abs(t) - im * t)
s = [1 0; 0 -1]
delta_t = 0.1
pt = uniTEMPO(s, delta_t, bcf, 1e-9)
pt2 = uniTEMPO([s], delta_t, t->bcf(t)*ones(1,1), 1e-9)
pt3 = uniTEMPO([s], delta_t, t->bcf(t)*ones(1,1), 1e-9; low_rank_svd=true, auto_nc=false, n_c=512)

# check trivial path accuracy
path = fill(1, 100)
ex_val = UniformTEMPO.exact_gaussian_influence(s, delta_t, bcf, path)
@test ex_val ≈ 1
@test isapprox(pt(path), 1, atol=1e-4)

# check single path accuracy
path = rand(1:4, 100)
ex_val = UniformTEMPO.exact_gaussian_influence(s, delta_t, bcf, path)
@test isapprox(pt(path), ex_val, atol=1e-4)
@test isapprox(pt(path), pt2(path), atol=1e-6)
@test isapprox(pt(path), pt3(path), atol=1e-6)

# check degeneracy filter
s_vals = rand(200)
s_vals[4] = s_vals[10] # add degeneracy 
filter, s_red = UniformTEMPO.degeneracy_filter(s_vals)
@test isapprox((transpose(s_red)*filter)[:], s_vals; atol=1e-8)
@test size(filter, 1) <= 199

# check commuting operator check
s1 = Float32.([1 0; 0 -1])
s2 = Float32.([10 0; 0 1])
s3 = s1
@test UniformTEMPO.check_commuting([s1 .+ 1f-6, s2, s3]; ftype=Float32)
@test !UniformTEMPO.check_commuting([s1 .+ 1f-4, s2, s3]; ftype=Float32)
s1 = ([0 10; -10 0]) * 1e5
s2 = ([0 10im; -10im 0]) * 1e5
@test UniformTEMPO.check_commuting([s1, s2 .+ 1e-13]; ftype=Float64)
@test !UniformTEMPO.check_commuting([s1, s2 .+ 1e-5]; ftype=Float64)