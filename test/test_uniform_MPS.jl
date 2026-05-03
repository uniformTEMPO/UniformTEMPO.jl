using UniformTEMPO
using Test

tol = 1e-12
d = 2
bd = 3
x = randn(ComplexF64, bd, d, bd)

L, xl = UniformTEMPO.left_orthogonalize(x)
@test isapprox((L*reshape(x, bd, d * bd))[:], (reshape(xl, bd * d, bd)*L)[:], atol=tol)

R, xr = UniformTEMPO.right_orthogonalize(x)
@test isapprox((reshape(x, bd * d, bd)*R)[:], (R*reshape(xr, bd, bd * d))[:], atol=tol)

v1, v2 = UniformTEMPO.fixed_points(x[:, 1, :])
@test transpose(v1) * v2 ≈ 1.0