# Canonical forms and SVD truncation for (unnormalized) uniform matrix product states according to [Vanderstraeten, Haegeman, Verstraetem. Tangent-space methods for uniform matrix product states. SciPost Phys. Lect. Notes 7 (2019)] and [Parker, Cao, Zaletel. Local matrix product operators: Canonical form, compression, and control theory. Phys. Rev. B 102, 035147 (2020)]

using LinearAlgebra
using TensorOperations
import KrylovKit.eigsolve


"""
    left_orthogonalize(x::AbstractArray{<:Number,3}; tol::Real=1e-14, maxIter::Real=1e4)

orthogonalization of uniform MPS using iterative QR (Ref [Phys. Rev. B 102, 035147 (2020)] Algorithm 3).

gauge transform `x` such that `xl * L = L * x` and `xl` is left-orthogonal.

Note that there exist pathological cases where this does not converge.
"""
function left_orthogonalize(x::AbstractArray{<:Number,3}; tol::Real=1e-14, maxIter::Real=1e4)
    n_iter = 1
    L = Matrix((1.0 + 0im) * I, size(x, 1), size(x, 1))
    convergence = 1.0
    xl = x
    while convergence > tol
        q, r = qr(reshape(xl, size(xl, 1) * size(xl, 2), :))
        q = Matrix(q)
        r = Matrix(r)
        q = reshape(q, size(xl, 1), size(xl, 2), :)
        @tensoropt xl[-1, -2, -3] := r[-1, 1] * q[1, -2, -3]
        L = r * L
        L /= norm(L)
        convergence = maximum(abs.(r' * r / tr(r' * r) * size(r, 2) - I))
        if n_iter > maxIter
            @warn "Warning, left decomposition has not converged $convergence"
            break
        end
        n_iter += 1
    end
    return L, xl
end


"""
    right_orthogonalize(x::AbstractArray{<:Number,3}; tol::Real=1e-14, maxIter::Real=1e4)

orthogonalization of uniform MPS using iterative QR (Ref [Phys. Rev. B 102, 035147 (2020)] Algorithm 3).

gauge transform `x` such that `R * xr = x * R` and `xr` is right-orthogonal.

Note that there exist pathological cases where this does not converge.
"""
function right_orthogonalize(x::AbstractArray{<:Number,3}; tol::Real=1e-14, maxIter::Real=1e4)
    n_iter = 1
    R = Matrix((1.0 + 0im) * I, size(x, 1), size(x, 1))
    convergence = 1.0
    xr = x
    while convergence > tol
        l, q = lq(reshape(xr, size(xr, 1), :))
        q = Matrix(q)
        l = Matrix(l)
        q = reshape(q, size(q,1), size(xr, 2), size(xr, 3))
        @tensoropt xr[-1, -2, -3] := q[-1, -2, 1] * l[1, -3]
        R = R * l
        R /= norm(R)
        convergence = maximum(abs.(l' * l / tr(l' * l) * size(l, 2) - I))
        if n_iter > maxIter
            @warn "Warning, right decomposition has not converged $convergence"
            break
        end
        n_iter += 1
    end

    return R, xr
end


"""
    mixed_canonical(x::AbstractArray{<:Number,3})


Find a mixed canonical form of a uniform mps `x`.
Returns left, center and right tensors as well as singular values.
"""
function mixed_canonical(x::AbstractArray{<:Number,3})
    L, xl = left_orthogonalize(x)
    R, xr = right_orthogonalize(x)
    u, s, v = svd(L * R)
    c = diagm(s)
    @tensoropt xl[-1 -2 -3] = u'[-1; 1] * xl[1 -2 2] * u[2 -3]
    @tensoropt xr[-1 -2 -3] = v[-1; 1] * xr[1 -2 2] * v'[2 -3]
    @tensoropt xc[-1 -2 -3] := xl[-1 -2 1] * c[1 -3]
    return xl, xc, xr, c
end;


"""
    fixed_points(a::AbstractMatrix{<:Number})

left and right fixed points of a square matrix `a`.
"""
function fixed_points(a::AbstractMatrix{<:Number})
    _, v_r, _ = eigsolve(a, 1, :LR)
    _, v_l, _ = eigsolve(transpose(a), 1, :LR)
    v_r = v_r[1]
    v_l = v_l[1] / (transpose(v_l[1]) * v_r)
    return v_r, v_l
end