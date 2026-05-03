# contains core uniTEMPO code.

using LinearAlgebra
using TensorOperations
using OMEinsum
using ProgressMeter
import Cubature.hcubature_v
import KrylovKit.eigsolve

"""
    compute_η(bcf::Function, delta_t::Real, n::Int)

Compute the discretized bath correlation function for the complex function `bcf` and `n` time steps `delta_t`.
"""
function compute_η(bcf::Function, delta_t::Real, n::Int)
    function bcf_re(t, v)
        @. v = real(bcf(t[1, :] - t[2, :]))
    end

    function bcf_im(t, v)
        @. v = imag(bcf(t[1, :] - t[2, :]))
    end

    function bcf_mod_re(t, v)
        @. v = real((t[1, :] - t[2, :]) |> t -> t > 0 ? bcf(t) : 0)
    end

    function bcf_mod_im(t, v)
        @. v = imag((t[1, :] - t[2, :]) |> t -> t > 0 ? bcf(t) : 0)
    end

    eta = zeros(ComplexF64, n + 1)
    for i in 1:n+1
        if i == 1
            # boundary term needs special care
            eta[i] = (hcubature_v(bcf_mod_re, [(i - 1) * delta_t, 0], [(i) * delta_t, delta_t], abstol=1e-7))[1]
            eta[1] += im * (hcubature_v(bcf_mod_im, [0, 0], [delta_t, delta_t], abstol=1e-7))[1]
        else
            eta[i] = (hcubature_v(bcf_re, [(i - 1) * delta_t, 0], [(i) * delta_t, delta_t], abstol=1e-7))[1]
            eta[i] += im * (hcubature_v(bcf_im, [(i - 1) * delta_t, 0], [(i) * delta_t, delta_t], abstol=1e-7))[1]
        end
    end
    return eta
end


"""
    trunc_svd_low_rank(M::AbstractMatrix{<:Number}, tol::Real, rank_cutoff::Int; truncation::Symbol=:rel, cap_rank::Int=100_000)

Low rank SVD truncation of matrix `M` with tolerance `tol` and maximum rank `rank_cutoff`. This is an optimized version of `trunc_svd` that first performs a low rank projection to allow for larger bond dimensions while keeping the computational cost manageable. The `truncation` and `cap_rank` parameters are passed to the second SVD step.
"""
function trunc_svd_low_rank(M::AbstractMatrix{<:Number}, tol::Real, rank_cutoff::Int; truncation::Symbol=:rel, cap_rank::Int=100_000)
    u, _, _ = psvd(M; rank=maximum([10, rank_cutoff])) # get low rank projection up to rank_cutoff, with minimum 10
    u2, s, vp = svd(u' * M; full=false) # get exact svd to maintain gauge
    u = u * u2 # rotate back
    rank_new = new_rank(s, tol, truncation; cap_rank=cap_rank)
    return u[:, 1:rank_new], s[1:rank_new], vp'[1:rank_new, :]
end


"""
    trunc_svd(M::AbstractMatrix{<:Number}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)

Truncated SVD of matrix `M` with tolerance `tol`. Returns the truncated U, S, V matrices. The `truncation` parameter can be `:rel` for relative truncation or `:abs` for absolute truncation. The `cap_rank` parameter sets an upper limit on the rank after truncation.
"""
function trunc_svd(M::AbstractMatrix{<:Number}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)
    u, s, vp = svd(M; full=false)
    rank_new = new_rank(s, tol, truncation; cap_rank=cap_rank)
    return u[:, 1:rank_new], s[1:rank_new], vp'[1:rank_new, :]
end


"""
    unitempo_iTEBD_step(contraction, i_tens::AbstractArray{<:Number,2}, A::AbstractArray{<:Number,3}, sAB::AbstractVector{<:Real}, B::AbstractArray{<:Number,3}, sBA::AbstractVector{<:Real}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000, low_rank_svd::Bool=false, svd_filtering_tol::Real=0)

single unitempo iTEBD step. tol is the SVD compression tolerance. `contraction` should be a callable that performs the "a,acx,x,xbd,d,bc->abcd" tensor contraction.
"""
function unitempo_iTEBD_step(contraction, i_tens::AbstractArray{<:Number,2}, A::AbstractArray{<:Number,3}, sAB::AbstractVector{<:Real}, B::AbstractArray{<:Number,3}, sBA::AbstractVector{<:Real}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000, low_rank_svd::Bool=false, svd_filtering_tol::Real=0)
    ftype = eltype(sAB)

    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)

    d1 = size(i_tens)[1]
    d2 = size(i_tens)[end]
    rank_BA = size(sBA)[1]

    if !iszero(svd_filtering_tol)
        # Cochin's iTEBD filtering algorithm from [https://arxiv.org/pdf/2603.06840]. Can be faster when the physical dimension is large.
        tol_pre = svd_filtering_tol # They use a smaller tolerance for the pre-svds, but not sure what to choose
        truncation_pre = :rel # they use relative truncation in the paper

        ui, si, vi = trunc_svd(transpose(i_tens), tol_pre; truncation=truncation_pre, cap_rank=cap_rank)

        # I absorb √sAB into ΘA and ΘB before the svd. I also absorb √si, I did not see any qualitative difference.

        @ein ΘA[x, sba, sab, si] := sBA[sba] * A[sba, x, sab] * sqrt.(sAB)[sab] * ui[x, si] * sqrt.(si)[si]
        uA, sA, vA = trunc_svd(reshape(ΘA, d2, :), tol_pre; truncation=truncation_pre, cap_rank=cap_rank)
        VA = reshape(diagm(sA) * vA, length(sA) * rank_BA, :)
        #VA = reshape(ein"ab,b->ab"(reshape(diagm(sA) * vA, :, length(si)), si), length(sA) * rank_BA, :) # include si in VA instead

        @ein ΘB[sab, si, sba, y] := sqrt.(sAB)[sab] * B[sab, y, sba] * sBA[sba] * vi[si, y] * sqrt.(si)[si]
        uB, sB, vB = trunc_svd(reshape(ΘB, :, d1), tol_pre; truncation=truncation_pre, cap_rank=cap_rank)
        VB = reshape(uB * diagm(sB), :, rank_BA * length(sB))

        Theta = reshape(permutedims(reshape(VA * VB, length(sA), rank_BA, rank_BA, length(sB)), (4, 2, 3, 1)), length(sB) * rank_BA, rank_BA * length(sA))

        if low_rank_svd
            u, s_vals, v = trunc_svd_low_rank(Theta, tol, trunc(Int, 1.5 * size(sAB, 1)); truncation=truncation, cap_rank=cap_rank)
        else
            u, s_vals, v = trunc_svd(Theta, tol; truncation=truncation, cap_rank=cap_rank)
        end

        u = reshape(permutedims(reshape(transpose(vB) * reshape(u, length(sB), :), d1, rank_BA, :), (2, 1, 3)), rank_BA * d1, :)
        v = reshape(permutedims(reshape(reshape(v, :, length(sA)) * transpose(uA), :, rank_BA, d2), (1, 3, 2)), :, d2 * rank_BA)
    else
        # MPS - gate contraction
        # @ein C[a, b, c, d] := sBA[a] * A[a, c, x] * sAB[x] * B[x, b, d] * sBA[d] * i_tens[b, c]
        C = contraction(Complex{ftype}.(sBA), Complex{ftype}.(A), Complex{ftype}.(sAB), Complex{ftype}.(B), Complex{ftype}.(sBA), Complex{ftype}.(i_tens))
        C = reshape(C, d1 * rank_BA, d2 * rank_BA)

        if low_rank_svd
            u, s_vals, v = trunc_svd_low_rank(C, tol, trunc(Int, 1.5 * size(sAB, 1)); truncation=truncation, cap_rank=cap_rank)
        else
            u, s_vals, v = trunc_svd(C, tol; truncation=truncation, cap_rank=cap_rank)
        end
    end

    rank_new = size(u, 2)
    u = reshape(u, size(sBA)[1], d1 * rank_new)
    v = reshape(v, rank_new * d2, rank_BA)

    # factor out sAB weights from A and B
    A = reshape(diagm(1 ./ sBA) * u, size(sBA)[1], d1, rank_new)
    B = reshape(v * diagm(1 ./ sBA), rank_new, d2, rank_BA)

    # new weights
    sAB = s_vals[1:rank_new]

    return A, sAB, B, sBA
end


"""
    degeneracy_filter(s_vals)

return degeneracy filter and unique values of the vector `s_vals`. transpose(uniquevals)*filter should give back the original s_diff vector. uniquevals is sorted by absolute value in descending order.
"""
function degeneracy_filter(s_vals::AbstractVector{<:Number})
    digits = max(0, trunc(Int, -log10(eps(eltype(s_vals)) * maximum(abs, s_vals))) - 3)
    s_vals_rounded = round.(s_vals, digits=digits) # round above numerical accuracy
    s_vals_rounded = replace(s_vals_rounded, -0.0 => 0.0)
    uniquevals = unique(s_vals_rounded)
    perm = sortperm(abs.(uniquevals))[end:-1:1]
    uniquevals = uniquevals[perm]
    filter = zeros(Bool, length(uniquevals), length(s_vals))
    for i in eachindex(s_vals)
        filter[argmin(abs.(uniquevals .- s_vals[i])), i] = true
    end
    for i in eachindex(uniquevals)
        uniquevals[i] = s_vals[argmin(abs.(uniquevals[i] .- s_vals))] # restore accurate values
    end
    return filter, uniquevals
end


"""
    get_boundaries_uniTEMPO(f0::AbstractMatrix{<:Number}, ftype::DataType)

Compute the left and right boundary vectors for uniTEMPO process tensor MPO `f0`. `ftype` specifies the data type used for the computation.
"""
function get_boundaries_uniTEMPO(f0::AbstractMatrix{<:Number}, ftype::DataType)
    if ftype == Float64
        get_boundaries_uniTEMPO(f0::AbstractMatrix{<:Number}, ftype::DataType, eigsolve_default_backend[])
    else
        get_boundaries_uniTEMPO(f0::AbstractMatrix{<:Number}, ftype::DataType, EigsolveGeneric())
    end
end
function get_boundaries_uniTEMPO(f0::AbstractMatrix{<:Number}, ftype::DataType, ::EigsolveKrylovKit)
    _, vecs_r, _ = eigsolve(transpose(f0), 1, :LR; maxiter=1000)
    _, vecs_l, _ = eigsolve((f0), 1, :LR; maxiter=1000)
    v_l = vecs_r[1]
    v_r = vecs_l[1] / (transpose(vecs_l[1]) * vecs_r[1])
    return v_l, v_r
end
function get_boundaries_uniTEMPO(f0::AbstractMatrix{<:Number}, ftype::DataType, ::EigsolveFullEDBackend)
    vr, vecs_r = eigen(transpose(f0))
    vl, vecs_l = eigen(f0)
    v_l = vecs_r[:, argmax(abs.(vr))]
    v_r = vecs_l[:, argmax(abs.(vl))]
    v_r = v_r / (transpose(v_l) * v_r)
    return v_l, v_r
end


"""
    check_commuting(s_ops::AbstractVector; ftype::DataType=Float64, tol::Real=eps(ftype) * 100)


Check if all operators in `s_ops` commute within tolerance `tol`.
"""
function check_commuting(s_ops::AbstractVector; ftype::DataType=Float64, tol::Real=eps(ftype) * 100)
    commuting = true
    scale = maximum(norm.(s_ops))
    if scale == 0
        return true
    end

    for s1 in s_ops, s2 in s_ops
        if norm(s1 * s2 - s2 * s1) > tol * scale^2
            commuting = false
            break
        end
    end
    return commuting
end


"""
    find_nc(bcf::Union{Function,Array}, tol::Real, delta_t::Real, s_vals::Vector, s_diff_red::Vector; n_c::Int=100_000, step::Real=2, n0::Int=2, truncation::Symbol=:rel, ftype::DataType=Float64)


Determine the memory cutoff `n_c` used in uniTEMPO automatically. `bcf` is the bath correlation function, `tol` is the SVD compression tolerance, `delta_t` is the Trotter time step, `s_vals` and `s_diff_red` are vectors containing the eigenvalues and filtered differences of eigenvalues of the coupling operator superoperator. The search starts at `n0` and increases by a factor of `step` until a maximum of `n_c` is reached.
"""
function find_nc(bcf::Union{Function,Array}, tol::Real, delta_t::Real, s_vals::Vector, s_diff_red::Vector; n_c::Int=100_000, step::Real=2, n0::Int=2, truncation::Symbol=:rel, ftype::DataType=Float64)
    # version for multiple coupling operators
    s_vals_l = [(s*ones(ftype, length(s))')[:] for s in s_vals]
    s_vals_r = [(ones(ftype, length(s))*s')[:] for s in s_vals]
    s_diff_red_dim = prod(length.(s_diff_red))
    s_dims = [length(s) for s in s_vals]
    m = length(s_vals)
    ν_dim = prod(s_dims .^ 2) + 1
    ind = CartesianIndices(Tuple(s_dims .^ 2))
    ind_diff = CartesianIndices(Tuple(length.(s_diff_red)))

    n = n0
    while n <= n_c
        i_tens = ones(Complex{ftype}, ν_dim, ν_dim)
        if typeof(bcf) <: Array
            cf = bcf[n, :, :]
        else
            cf = Complex{ftype}.(delta_t^2 * bcf(n * delta_t))
        end

        for i in 1:s_diff_red_dim, j in 1:(ν_dim-1)
            idxsi = ind_diff[i]
            idxsj = ind[j]
            sd_t = [s_diff_red[k][idxsi[k]] for k in 1:m]
            s_ls = [s_vals_l[k][idxsj[k]] for k in 1:m]
            s_rs = [s_vals_r[k][idxsj[k]] for k in 1:m]
            i_tens[j, i] = exp(-sd_t' * (cf * s_ls - conj.(cf) * s_rs))
        end

        _, s_vals, _ = svd(i_tens)
        rank = new_rank(s_vals, tol, truncation)

        if rank == 1
            return n
        end

        n_new = trunc(Int, n * step)
        if n_new == n
            n_new += 1
        end
        n = n_new

        if (n > n_c && n < n_c * step)
            n = n_c
        end
    end
    throw(ErrorException("Could not find a memory cutoff within the maximum limit $n_c. Consider increasing n_c or lowering the tolerance. Your bath correlation function may decay too slowly."))
end


"""
    uniTEMPO(s::Union{AbstractMatrix{<:Number}, Vector}, delta_t::Real, bcf::Union{Function, Array}, tol::Real; auto_nc::Bool=true, n_c::Int=100_000, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, low_rank_svd::Bool=false, svd_filtering_tol::Real=0, max_rank::Int=100_000)

Compute the process tensor for a Gaussian bath using the uniTEMPO algorithm. Returns a `UniformPTMPO` representing the compressed process tensor.

# Arguments
- `s` is the coupling operator (must be hermitian) or a list of coupling operators.
- `delta_t` is the Trotter time step.
- `bcf` is the bath correlation function.
- `tol` is the SVD compression tolerance, determining the accuracy of the compression.

# Keyword Arguments
- `auto_nc` if set to true, a memory cutoff is determined automatically based on the decay of the bath correlation function and the provided tolerance (recommended). Otherwise, the user can specify `n_c` directly.
- `n_c` is the maximum memory cutoff. Note that auto_nc will not search beyond this value.
- `truncation` specifies the truncation scheme for singular value truncation. It can be `:rel` for relative, `:abs` for absolute, see function `new_rank`.
- `cap_rank` fixes the maximum bond dimension allowed during the SVD compression. Usually it is not recommended to use this option.
- `max_rank` is a hard limit on the bond dimension. The algorithm will throw an error if this limit is reached. 
- `low_rank_svd` if set to true, compute the low-rank subspace first and then perform SVD on the low-rank subspace only. This can be beneficial for simulations with large system dimensions. It is recommended to set `truncation` to `:abs` when using this option to avoid issues with determining the rank based on the relative cutoff when the full singular value spectrum is not computed.
- `svd_filtering_tol` if set to a nonzero value, use an iTEBD scheme with SVD filtering on the physical indices. The extra filtering is performed with the given tolerance. 
"""
function uniTEMPO(s::AbstractMatrix{<:Number}, delta_t::Real, bcf::Union{Function,Vector}, tol::Real; auto_nc::Bool=true, n_c::Int=100_000, truncation::Symbol=:rel, cap_rank::Int=100_000, low_rank_svd::Bool=false, svd_filtering_tol::Real=0, ftype::DataType=Float64, max_rank::Int=100_000)
    # this is the version for a single coupling operator, the version for multiple coupling operators is below. 
    # TODO: Merge it with multiple coupling version. Currently multi coupling is still slower due to some extra overhead (builing i_tens).
    @assert s' == s "`s` must be hermitian"

    s_vals, u = eigen(Hermitian(s))
    p, s_vals = degeneracy_filter(s_vals)
    u_super = Matrix(kron(transpose(u'), u))
    @tensor filter[-1, -2, -3, -4] := p[-1, -3] * p[-2, -4]
    filter = reshape(filter, :, size(s, 1)^2)
    s_dim = length(s_vals)

    s_diff = zeros(ftype, s_dim^2 + 1)
    s_sum = zeros(ftype, s_dim^2 + 1)
    for ν in 1:(s_dim^2)
        i, j = 1 + Int(floor((ν - 1) / s_dim)), 1 + (ν - 1) % s_dim
        s_diff[ν] = s_vals[j] - s_vals[i]
        s_sum[ν] = s_vals[j] + s_vals[i]
    end

    s_diff_filter, s_diff_red = degeneracy_filter(s_diff) # filter differences of s_vals on backwards passing to reduce network size

    if typeof(bcf) <: Vector
        n_c = minimum([length(bcf), n_c])
    end

    if auto_nc
        if typeof(bcf) <: Vector
            n_c = find_nc(reshape(bcf, :, 1, 1), tol, delta_t, [s_vals], [s_diff_red]; n_c=n_c, truncation=truncation, ftype=ftype)
        else
            n_c = find_nc(t -> bcf(t) * ones(1, 1), tol, delta_t, [s_vals], [s_diff_red]; n_c=n_c, truncation=truncation, ftype=ftype)
        end
    end
    @show n_c

    if typeof(bcf) <: Vector
        η = Complex{ftype}.(bcf[1:n_c])
    else
        η = Complex{ftype}.(compute_η(bcf, delta_t, n_c))
    end

    A = ones(Complex{ftype}, 1, length(s_diff_red), 1)
    B = ones(Complex{ftype}, 1, length(s_sum), 1)
    sAB = Vector(ones(ftype, 1))
    sBA = Vector(ones(ftype, 1))
    rank_is_one = true

    contraction = ein"a,acx,x,xbd,d,bc->abcd"

    optimized_contraction = optimize_code(contraction, uniformsize(contraction, length(s_sum)), GreedyMethod())

    p = ProgressUnknown(desc="building the influence matrix", spinner=true)

    for k in 0:(n_c-2)
        i_tens = exp.(-real(η[n_c-k]) * (s_diff * s_diff_red') - im * imag(η[n_c-k]) * (s_sum * s_diff_red'))

        #A_new, sAB_new, B_new, sBA_new 
        B, sBA, A, sAB = unitempo_iTEBD_step(optimized_contraction, i_tens, A, sAB, B, sBA, tol; truncation=truncation, low_rank_svd=low_rank_svd, cap_rank=cap_rank, svd_filtering_tol=svd_filtering_tol)

        if rank_is_one
            if all([length(sAB) == 1, length(sBA) == 1])
                # reset to initial mps if rank is still one
                A = ones(Complex{ftype}, 1, length(s_diff_red), 1)
                B = ones(Complex{ftype}, 1, length(s_sum), 1)
                sAB = Vector(ones(ftype, 1))
                sBA = Vector(ones(ftype, 1))
            else
                rank_is_one = false
                if !auto_nc
                    (k == 0) && (@warn "The memory cutoff n_c may be too small for the given tol value. The algorithm may become unstable and inaccurate. It is recommended to increase n_c until this message does no longer appear.")
                end
            end
        end

        if maximum([length(sAB), length(sBA)]) > max_rank
            throw("Algorithm reached bond dimension limit $(max_rank).")
        end

        next!(p, spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", showvalues=[("remaining steps", n_c - k - 1), ("current bond dimension", maximum([length(sAB), length(sBA)]))])
    end

    finish!(p)

    # the last step does not require svd compression
    i_tens = exp.(-real(η[1]) * (s_diff .* s_diff) - im * imag(η[1]) * (s_sum .* s_diff))

    @tensor Anew[a, c, x] := A[a, f, x] * s_diff_filter[f, c] # restore s_diff degeneracy

    @ein f[b, c, a] := Anew[a, c, x] * sAB[x] * B[x, c, b] * sBA[b] * i_tens[c]

    if (size(f, 1)[1] == 1)
        println("bond dimension 1 (trivial influence functional)")
        return UniformPTMPO(size(s, 1), delta_t)
    end

    @ein ut[-1, -2, -3] := u_super[-1, 2] * filter[-2, 2] * Matrix(u_super')[2, -3]
    @tensoropt q[-1, -2, -3, -4] := f[:, 1:s_dim^2, :][-1, 1, -3] * ut[-2, 1, -4]

    v_l, v_r = get_boundaries_uniTEMPO(f[:, end, :], ftype)

    return UniformPTMPO(size(s, 1), delta_t, q, v_r, transpose(v_l))
end
function uniTEMPO(s_ops::AbstractVector, delta_t::Real, bcf::Union{Function,Array}, tol::Real; auto_nc::Bool=true, n_c::Int=100_000, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, low_rank_svd::Bool=false, svd_filtering_tol::Real=0, max_rank::Int=100_000)
    s_vals = []
    basis = []
    filter = []
    filter_s_diff = []
    s_diff_red = []

    for s in s_ops
        @assert size(s, 1) == size(s, 2) "`s` must be square matrices"
        @assert size(s_ops[1]) == size(s) "All coupling operators must have the same dimension"
        @assert s' == s "All coupling operators must be hermitian"
        s_vals_, u = eigen(Hermitian(s))
        p, s_vals_ = degeneracy_filter(s_vals_)
        u_super = Matrix(kron(transpose(u'), u))
        push!(s_vals, s_vals_)
        @tensor proj_super[-1, -2, -3, -4] := p[-1, -3] * p[-2, -4]
        push!(filter, reshape(proj_super, :, size(s_ops[1], 1)^2))
        filter_s_diff_, s_diff_red_ = degeneracy_filter((s_vals_*ones(length(s_vals_))'.-ones(length(s_vals_))*s_vals_')[:])
        push!(filter_s_diff, filter_s_diff_)
        push!(s_diff_red, s_diff_red_)
        push!(basis, u_super)
    end

    s_diff_red_dim = prod(length.(s_diff_red))
    s_dims = [length(s) for s in s_vals]

    # check if coupling operators commute to decide trotter scheme
    commuting = check_commuting(s_ops; ftype=ftype)
    delta_t_orig = deepcopy(delta_t)

    if !commuting
        delta_t /= 2 # half time step for second order trotter
        n_c *= 2
    end

    if typeof(bcf) <: Array
        n_c = minimum([size(bcf, 1), n_c])
        if !commuting
            @warn "The coupling opertors are non-commuting which changes the time step internally by a factor of one half. Since you provided the bcf as an array, make sure that you used half of the time step delta_t for your bcf computation."
        end
    end

    if auto_nc
        n_c = find_nc(bcf, tol, delta_t, s_vals, s_diff_red; n_c=n_c, truncation=truncation, ftype=ftype)
    end
    @show n_c

    if typeof(bcf) <: Array
        η = Complex{ftype}.((bcf[1:n_c, :, :]))
    else
        m = size(bcf(0), 1)
        η = zeros(Complex{ftype}, n_c + 1, m, m)
        for i in axes(η, 2), j in axes(η, 3)
            # TODO: make bcf a matrix of functions rather than evaluating the full bcf matrix at each time step and then integrating.
            η[:, i, j] = Complex{ftype}.(compute_η(t -> bcf(t)[i, j], delta_t, n_c))
        end
    end

    @assert length(s_ops) == size(η, 2) "Number of coupling operators and size of bath correlation functions must match."
    m = size(η, 2)

    s_vals_l = [(s*ones(ftype, length(s))')[:] for s in s_vals]
    s_vals_r = [(ones(ftype, length(s))*s')[:] for s in s_vals]

    ν_dim = prod(s_dims .^ 2) + 1
    A = ones(Complex{ftype}, 1, s_diff_red_dim, 1)
    B = ones(Complex{ftype}, 1, ν_dim, 1)
    sAB = Vector(ones(ftype, 1))
    sBA = Vector(ones(ftype, 1))
    rank_is_one = true

    contraction = ein"a,acx,x,xbd,d,bc->abcd"

    optimized_contraction = optimize_code(contraction, uniformsize(contraction, ν_dim), GreedyMethod())

    p = ProgressUnknown(desc="building the influence matrix", spinner=true)
    ind = CartesianIndices(Tuple(s_dims .^ 2))
    ind_diff = CartesianIndices(Tuple(length.(s_diff_red)))

    for k in 0:(n_c-2)
        i_tens = ones(Complex{ftype}, ν_dim, s_diff_red_dim)

        cf = η[n_c-k, :, :]
        for i in 1:s_diff_red_dim, j in 1:(ν_dim-1)
            idxsi = ind_diff[i]
            idxsj = ind[j]
            sd_t = [s_diff_red[k][idxsi[k]] for k in 1:m]
            s_ls = [s_vals_l[k][idxsj[k]] for k in 1:m]
            s_rs = [s_vals_r[k][idxsj[k]] for k in 1:m]
            i_tens[j, i] = exp(-sd_t' * (cf * s_ls - conj.(cf) * s_rs))
        end

        # A_new, sAB_new, B_new, sBA_new 
        B, sBA, A, sAB = unitempo_iTEBD_step(optimized_contraction, i_tens, A, sAB, B, sBA, tol; truncation=truncation, low_rank_svd=low_rank_svd, cap_rank=cap_rank, svd_filtering_tol=svd_filtering_tol)

        if rank_is_one
            if all([length(sAB) == 1, length(sBA) == 1])
                # reset to initial mps if rank is still one
                A = ones(Complex{ftype}, 1, s_diff_red_dim, 1)
                B = ones(Complex{ftype}, 1, ν_dim, 1)
                sAB = Vector(ones(ftype, 1))
                sBA = Vector(ones(ftype, 1))
            else
                rank_is_one = false
                if !auto_nc
                    ((k == 0) && (@warn "The memory cutoff n_c may be too small for the given rtol value. The algorithm may become unstable and inaccurate. It is recommended to increase n_c until this message does no longer appear."))
                end
            end
        end

        if maximum([length(sAB), length(sBA)]) > max_rank
            throw("Algorithm reached bond dimension limit $(max_rank).")
        end

        next!(p, spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", showvalues=[("remaining steps", n_c - k - 1), ("current bond dimension", maximum([length(sAB), length(sBA)]))])
    end

    finish!(p)

    A0 = A[:, end, :] # last entry of A corresponds to all s_vals = 0 because s_diff_red always has zero at the end
    B0 = B[:, end, :] # padded last entry of B corresponds to all s_vals = 0
    A = reshape(A, size(A, 1), [length(s_diff_red[k]) for k in eachindex(s_diff_red)]..., size(A, 3))
    A = ncon([A, filter_s_diff...], [[-1, (1:m)..., -(m + 2)], [[k, -k - 1] for k in 1:m]...]) # apply s_diff filter to restore s_diff degeneracy
    A = reshape(A, size(sBA, 1), :, size(sAB, 1))
    B = B[:, 1:end-1, :] # discard "zero" entry of B

    # the last step does not require svd compression
    i_tens = ones(Complex{ftype}, ν_dim - 1)
    i_tens_o = ones(Complex{ftype}, ν_dim - 1)
    i_tens_e = ones(Complex{ftype}, ν_dim - 1)

    cf = η[1, :, :]

    # trotter corrections for even and odd time steps
    cf2 = deepcopy(cf)
    for i in 1:m, j in 1:m
        if i != j
            cf2[i, j] = cf[i, j] + conj(cf[j, i])
        end
    end
    R_e = (1:m) .<= (1:m)'
    R_o = (1:m) .>= (1:m)'

    for i in 1:(ν_dim-1)
        idxsi = ind[i]
        s_l = [s_vals_l[k][idxsi[k]] for k in 1:m]
        s_r = [s_vals_r[k][idxsi[k]] for k in 1:m]
        i_tens[i] = exp(-(s_l - s_r)' * (cf * s_l - conj.(cf) * s_r))
        i_tens_e[i] = exp((s_l' * conj.(cf) * s_r) + (s_r' * cf * s_l) - (s_l' * (R_e .* cf2) * s_l) - (s_r' * (R_e .* conj.(cf2)) * s_r))
        i_tens_o[i] = exp((s_l' * conj.(cf) * s_r) + (s_r' * cf * s_l) - (s_l' * (R_o .* cf2) * s_l) - (s_r' * (R_o .* conj.(cf2)) * s_r))
    end

    bt = [(@ein t_[-1, -2, -3] := basis[n][-1, 2] * (filter[n])[-2, 2] * Matrix(basis[n]')[2, -3]) for n in eachindex(basis)]

    @ein f0[b, a] := A0[a, x] * sAB[x] * B0[x, b] * sBA[b]

    if commuting
        @ein f[b, c, a] := A[a, c, x] * sAB[x] * B[x, c, b] * sBA[b] * i_tens[c]
        f = reshape(f, size(f, 1), (s_dims .^ 2)..., size(f, 3))
        if length(basis) == 1
            @tensoropt q[-1, -2, -3, -4] := (f[:, 1:((s_dims[1])^2), :])[-2, 1, -4] * (bt[1])[-1, 1, -3]
        else
            q = ncon([f, bt...], [[-2, (1:m)..., -4], [-1, 1, m + 1], [[m + i - 1, i, m + i] for i in 2:(m-1)]..., [m + m - 1, m, -3]])
        end
        q = permutedims(q, (2, 1, 4, 3))
    else
        @ein fe[b, c, a] := A[a, c, x] * sAB[x] * B[x, c, b] * sBA[b] * i_tens_e[c]
        @ein fo[b, c, a] := A[a, c, x] * sAB[x] * B[x, c, b] * sBA[b] * i_tens_o[c]
        # inverting the ordering on even and odd steps to ensure symmetric trotterization
        fe = reshape(fe, size(fe, 1), (s_dims .^ 2)..., size(fe, 3))
        fo = reshape(fo, size(fo, 1), (s_dims .^ 2)..., size(fo, 3))
        qe = ncon([fe, bt...], [[-2, (1:m)..., -4], [-1, 1, m + 1], [[m + i - 1, i, m + i] for i in 2:(m-1)]..., [m + m - 1, m, -3]])
        qo = ncon([fo, bt...], [[-2, (1:m)..., -4], [m + m - 1, 1, -3], [[m + m - i, i, m + m + 1 - i] for i in 2:(m-1)]..., [-1, m, m + 1]])
        # final double-step propagator in second order trotter
        @tensor q[-1, -2, -3, -4] := qe[-2, -1, 1, 2] * qo[1, 2, -4, -3]
    end

    if (size(f0, 1) == 1)
        println("bond dimension 1 (trivial influence functional)")
        return UniformPTMPO(size(s_ops[1], 1), delta_t_orig)
    end

    # boundaries
    v_l, v_r = get_boundaries_uniTEMPO(f0, ftype)

    return UniformPTMPO(size(s_ops[1], 1), delta_t_orig, q, v_r, transpose(v_l))
end


"""
    exact_gaussian_influence(s::AbstractMatrix{<:Number}, delta_t::Real, bcf::Function, ν_path::AbstractVector{<:Int})

Evaluate the exact Gaussian influence functional along a given path. This only works in the case of a single diagonal coupling operator. Can be used as a benchmark to check convergence.
"""
function exact_gaussian_influence(s::AbstractMatrix{<:Number}, delta_t::Real, bcf::Function, ν_path::AbstractVector{<:Int})
    @assert s' == s "coupling operator `s` must be hermitian"
    @assert s == diagm(diag(s)) "coupling operator `s` must be diagonal for this function to work."
    s_vals = diag(s)
    s_dim = length(s_vals)
    ν_dim = s_dim^2 + 1
    η = compute_η(bcf, delta_t, length(ν_path) + 1)
    s_diff = zeros(ComplexF64, ν_dim)
    s_sum = zeros(ComplexF64, ν_dim)
    for ν in 1:(ν_dim-1)
        i, j = 1 + Int(floor((ν - 1) / s_dim)), 1 + (ν - 1) % s_dim
        s_diff[ν] = s_vals[j] - s_vals[i]
        s_sum[ν] = s_vals[j] + s_vals[i]
    end

    n = length(ν_path)

    # this computes the exact value
    s_diff_path = [s_diff[ν_path[i]] for i in eachindex(ν_path)]
    s_sum_path = [s_sum[ν_path[i]] for i in eachindex(ν_path)]
    exponent = ComplexF64(0)
    for s in 1:n, v in 1:s
        exponent += -s_diff_path[s] * real(η[1+s-v]) * s_diff_path[v] - im * s_diff_path[s] * imag(η[1+s-v]) * s_sum_path[v]
    end

    return exp(exponent)
end