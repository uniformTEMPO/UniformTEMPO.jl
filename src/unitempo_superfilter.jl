# uniTEMPO with the reduced ("super filtered") network: both legs of the iTEBD are carried in bases of quadratic
# polynomials of the bath-channel coordinates of the eigenvalue labels, so the bulk does not depend on the system
# dimension. The memory gates are applied in their second-order Taylor form, and the process tensor is returned with the
# short-time propagator as a linear operator (SparseUniformPTMPO), without forming the dense q.


"""
    SparsePropagator

Short-time propagator of a uniform process tensor as a linear operator: `q * X` maps the state `X[a, s]` (bond index ×
vectorized density matrix) through the bath layers of one time step, each

    X'[b, s'] = Σ_{x,e} B[b, (x, e)] Σ_{c,s} S[(c, s), (e, s')] Σ_a A[(x, c), a] X[a, s],

with `A`, `B` the iTEBD tensors in the reduced bases and `S` the superoperators of the reduced legs of that layer.
"""
struct SparsePropagator{T<:Number}
    A::Matrix{T}
    B::Matrix{T}
    S::Vector{Matrix{T}}
end
function Base.:*(q::SparsePropagator, X::AbstractMatrix)
    for S in q.S
        X = q.B * reshape(reshape(q.A * X, :, size(S, 1)) * S, size(q.B, 2), :)
    end
    return X
end


"""
    SparseUniformPTMPO

Uniform process tensor as [`UniformPTMPO`](@ref), with the short-time propagator `q` given as a [`SparsePropagator`](@ref)
instead of a dense array. Returned by [`uniTEMPO_superfilter`](@ref).
"""
struct SparseUniformPTMPO
    s_dim::Int
    delta_t::Real
    q::SparsePropagator
    v_r::AbstractVector{<:Number}
    v_l::Transpose{<:Number,<:AbstractVector}
end
bond_dim(pt::SparseUniformPTMPO) = size(pt.q.B, 1)


"""
    UniformPTMPO(pt::SparseUniformPTMPO)

Dense process tensor with the same bond dimension and boundaries: the short-time propagator `q[b, s', a, s]` is the
product of the dense layer maps M[(b, s'), (a, s)] = Σ_{x,e,c} B[b, (x, e)] S[(c, s), (e, s')] A[(x, c), a], each built
with two matrix products (bond tensors first, then the layer superoperator), using no more memory than `q` itself. All functions of a [`UniformPTMPO`](@ref) (steady state, correlation functions, ...) are then
available; the dense `q` has (χ d²)² entries, which is the memory the sparse form avoids.
"""
function UniformPTMPO(pt::SparseUniformPTMPO)
    χ, d2 = bond_dim(pt), pt.s_dim^2
    A, B = pt.q.A, pt.q.B
    Rd, R = size(pt.q.S[1], 1) ÷ d2, size(pt.q.S[1], 2) ÷ d2   # S is (c, s) × (e, s')
    x = size(A, 1) ÷ Rd                                        # A is (x, c) × a, B is b × (x, e)
    Q = nothing   # product of the layer maps (one layer for commuting couplings: no product needed)
    for S in pt.q.S
        # C[(b, a), (c, e)] = Σ_x B[b, (x, e)] A[(x, c), a]: the bond tensors contracted first (small, R R_d χ² entries)
        Bm = reshape(permutedims(reshape(B, χ, x, R), (1, 3, 2)), χ * R, x)
        C = reshape(permutedims(reshape(Bm * reshape(A, x, Rd * χ), χ, R, Rd, χ), (1, 4, 3, 2)), χ * χ, Rd * R)
        # M[(b, a), (s, s')] = Σ_{c,e} C[(b, a), (c, e)] S[(c, s), (e, s')]
        Sm = reshape(permutedims(reshape(S, Rd, d2, R, d2), (1, 3, 2, 4)), Rd * R, d2 * d2)
        M = permutedims(reshape(C * Sm, χ, χ, d2, d2), (1, 4, 2, 3))          # [b, s', a, s]
        Q = isnothing(Q) ? reshape(M, χ * d2, χ * d2) : reshape(M, χ * d2, χ * d2) * Q
    end
    return UniformPTMPO(pt.s_dim, pt.delta_t, reshape(Q, χ, d2, χ, d2), pt.v_r, pt.v_l)
end


"""
    evolve(pt::SparseUniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s=nothing, return_full::Bool=false)

Time evolution with a [`SparseUniformPTMPO`](@ref), as for a [`UniformPTMPO`](@ref).
"""
function evolve(pt::SparseUniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing, return_full::Bool=false)
    dt, d = pt.delta_t, pt.s_dim
    if h_s isa Function
        half_steps = i -> (transpose(local_channel((dt * (i - 1), dt * (i - 0.5)), h_s)), transpose(local_channel((dt * (i - 0.5), dt * i), h_s)))
    else
        h_s = isnothing(h_s) ? zeros(d, d) : h_s
        u = transpose(kron(transpose(exp(im * h_s * dt / 2)), exp(-im * h_s * dt / 2)))
        half_steps = i -> (u, u)
    end
    state = pt.v_r * transpose(ρ_0[:])

    ρ_t = Vector{Matrix{ComplexF64}}(undef, n + 1)
    ρ_t[1] = reshape(pt.v_l * state, d, d)

    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        u1, u2 = half_steps(i)
        state = (pt.q * (state * u1)) * u2
        ρ_t[i+1] = reshape(pt.v_l * state, d, d)
    end

    return_full && (return state)
    return ρ_t
end


# monomials of degree ≤ 2 of the columns of y: 1, y_a, y_a y_b (all ordered pairs, in kron order)
quadratic_monomials(y::AbstractMatrix) = hcat(ones(eltype(y), size(y, 1)), y, reshape(reshape(y, size(y, 1), :, 1) .* reshape(y, size(y, 1), 1, :), size(y, 1), :))

# orthonormal basis of the column span
orthonormal_columns(M::AbstractMatrix) = (F = svd(M); F.U[:, 1:count(>(1e-12 * F.S[1]), F.S)])


"""
    superfilter_iTEBD_step(gate::AbstractMatrix{<:Number}, R::Int, R_d::Int, A, sAB, B, sBA, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000, low_rank_svd::Bool=false)

Single uniTEMPO iTEBD step in the reduced bases: `B` carries the path index (dimension `R`) and `A` the future index
(dimension `R_d`), and the reduced gate is the matrix `gate[(e, c'), (b, c)]` on both.
"""
function superfilter_iTEBD_step(gate::AbstractMatrix{<:Number}, R::Int, R_d::Int, A::AbstractArray{<:Number,3}, sAB::AbstractVector{<:Real}, B::AbstractArray{<:Number,3}, sBA::AbstractVector{<:Real}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000, low_rank_svd::Bool=false)
    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)
    rank_BA, rank_AB = length(sBA), length(sAB)

    # T[b, c, a, d] = sBA[a] A[a, c, x] sAB[x] B[x, b, d] sBA[d], the gate acts on (b, c)
    As = reshape(sBA .* reshape(A, rank_BA, :), rank_BA * R_d, rank_AB) .* transpose(sAB)
    Bs = reshape(reshape(B, :, rank_BA) .* transpose(sBA), rank_AB, R * rank_BA)
    T = permutedims(reshape(As * Bs, rank_BA, R_d, R, rank_BA), (3, 2, 1, 4))
    Y = reshape(gate * reshape(T, R * R_d, :), R, R_d, rank_BA, rank_BA)
    C = reshape(permutedims(Y, (3, 1, 2, 4)), rank_BA * R, R_d * rank_BA)

    if low_rank_svd
        u, s_vals, v = trunc_svd_low_rank(C, tol, trunc(Int, 1.5 * rank_AB); truncation=truncation, cap_rank=cap_rank)
    else
        u, s_vals, v = trunc_svd(C, tol; truncation=truncation, cap_rank=cap_rank)
    end

    # factor out sBA weights
    rank_new = size(u, 2)
    A = reshape(Diagonal(1 ./ sBA) * reshape(u, rank_BA, R * rank_new), rank_BA, R, rank_new)
    B = reshape(reshape(v, rank_new * R_d, rank_BA) * Diagonal(1 ./ sBA), rank_new, R_d, rank_BA)
    return A, s_vals, B, sBA
end


"""
    uniTEMPO_superfilter(s::Union{AbstractMatrix{<:Number}, Vector}, delta_t::Real, bcf::Union{Function, Array}, tol::Real; auto_nc::Bool=true, n_c::Int=100_000, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, low_rank_svd::Bool=false, max_rank::Int=100_000)

Compute the process tensor for a Gaussian bath using the uniTEMPO algorithm with the reduced network: the path index
and the future index of the iTEBD are carried in bases of polynomials of degree two in the bath-channel coordinates of
the eigenvalue labels, and the memory gates in their second-order Taylor form. The bulk dimensions are set by the
number of bath channels `r` (`R = (r + 1)(2r + 1)` on the path index) and not by the system dimension. The error of the
reduction is of second order in the time step, as the Trotter splitting. Returns a [`SparseUniformPTMPO`](@ref).

The arguments and keyword arguments are those of [`uniTEMPO`](@ref).
"""
function uniTEMPO_superfilter(s::AbstractMatrix{<:Number}, delta_t::Real, bcf::Union{Function,Vector}, tol::Real; kwargs...)
    return uniTEMPO_superfilter([s], delta_t, bcf isa Vector ? reshape(bcf, :, 1, 1) : t -> bcf(t) * ones(1, 1), tol; kwargs...)
end
function uniTEMPO_superfilter(s_ops::AbstractVector, delta_t::Real, bcf::Union{Function,Array}, tol::Real; auto_nc::Bool=true, n_c::Int=100_000, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, low_rank_svd::Bool=false, max_rank::Int=100_000)
    s_dim = size(s_ops[1], 1)
    m = length(s_ops)
    s_vals = []
    projectors = []
    filter_s_diff = []
    s_diff_red = []

    for s in s_ops
        @assert size(s) == (s_dim, s_dim) "All coupling operators must be square and of the same dimension"
        @assert s' == s "All coupling operators must be hermitian"
        s_vals_, u = eigen(Hermitian(Matrix(s)))
        p, s_vals_ = degeneracy_filter(s_vals_)
        push!(s_vals, s_vals_)
        push!(projectors, [u * Diagonal(Vector{ftype}(p[i, :])) * u' for i in axes(p, 1)])
        filter_s_diff_, s_diff_red_ = degeneracy_filter((s_vals_*ones(length(s_vals_))'.-ones(length(s_vals_))*s_vals_')[:])
        push!(filter_s_diff, filter_s_diff_)
        push!(s_diff_red, s_diff_red_)
    end

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
        η = zeros(Complex{ftype}, n_c + 1, m, m)
        for i in axes(η, 2), j in axes(η, 3)
            η[:, i, j] = Complex{ftype}.(compute_η(t -> bcf(t)[i, j], delta_t, n_c))
        end
    end
    @assert m == size(η, 2) "Number of coupling operators and size of bath correlation functions must match."

    # eigenvalue labels: path index μ = (ν_1, ..., ν_m), ν_l = i + r_l (j - 1) for the forward (i) and backward (j)
    # eigenvalues of s_l, and the (degeneracy filtered) eigenvalue differences d of the future index
    r = length.(s_vals)
    ind = CartesianIndices(Tuple(r .^ 2))
    ind_diff = CartesianIndices(Tuple(length.(s_diff_red)))
    D = length(ind)
    s_l = [s_vals[l][1+(ind[μ][l]-1)%r[l]] for l in 1:m, μ in 1:D]
    s_r = [s_vals[l][1+(ind[μ][l]-1)÷r[l]] for l in 1:m, μ in 1:D]
    s_d = [s_diff_red[l][ind_diff[d][l]] for l in 1:m, d in 1:length(ind_diff)]
    diff_index = [LinearIndices(ind_diff)[CartesianIndex(Tuple(findfirst(filter_s_diff[l][:, ind[μ][l]]) for l in 1:m))] for μ in 1:D]
    d0 = findfirst(d -> all(iszero, s_d[:, d]), axes(s_d, 2))

    # bath-channel coordinates: η_k = X_k Q† with Q spanning the rows of all η_k, and the exponent of the gate
    # -Δᵀ(η_k s^l - η_k* s^r) = wᵀ T_k ζ with ζ = (Q† s^l, Qᵀ s^r) and w = Pᵀ Δ, P spanning the columns of all X_k, X_k*
    F = svd(reduce(vcat, [η[k, :, :] for k in 2:n_c]))
    Q = F.V[:, 1:count(>(1e-10 * F.S[1]), F.S)]
    X = [η[k, :, :] * Q for k in axes(η, 1)]
    F = svd(reduce(hcat, [hcat(x, conj.(x)) for x in X[2:n_c]]))
    P = F.U[:, 1:count(>(1e-10 * F.S[1]), F.S)]
    ζ = vcat(hcat(transpose(s_l) * conj.(Q), transpose(s_r) * Q), zeros(1, 2 * size(Q, 2))) # trivial index ν = D + 1 at zero
    w = transpose(s_d) * P

    # bases: quadratic polynomials in ζ for the path index, and in w for the future index with the zero difference kept
    # exact (first column), and the monomials projected onto them
    Mz = quadratic_monomials(ζ)
    Mw = quadratic_monomials(w)
    v = orthonormal_columns(Mz)
    Mw0 = copy(Mw)
    Mw0[d0, :] .= 0
    u = hcat((1:size(w, 1)) .== d0, orthonormal_columns(Mw0))
    R, R_d = size(v, 2), size(u, 2)
    Pz = permutedims(reduce(hcat, [vec(v' * (Mz[:, j] .* v)) for j in axes(Mz, 2)])) # Pz[j, (e, b)]
    Pw = permutedims(reduce(hcat, [vec(u' * (Mw[:, i] .* u)) for i in axes(Mw, 2)])) # Pw[i, (c', c)]

    # reduced gate of η_k: exp(wᵀ T ζ) to second order, 1 + wᵀ T ζ + (wᵀ T ζ)² / 2, in the monomials
    function reduced_gate(k)
        T = hcat(-P' * X[k], P' * conj.(X[k]))
        coef = cat(ones(1, 1), T, kron(T, T) / 2; dims=(1, 2))
        G = transpose(Pz) * transpose(coef) * Pw
        return reshape(permutedims(reshape(G, R, R, R_d, R_d), (1, 3, 2, 4)), R * R_d, R * R_d)
    end

    A = reshape(u' * ones(Complex{ftype}, size(u, 1)), 1, R_d, 1)
    B = reshape(v' * ones(Complex{ftype}, D + 1), 1, R, 1)
    sAB = Vector(ones(ftype, 1))
    sBA = Vector(ones(ftype, 1))
    A_init, B_init = copy(A), copy(B)
    rank_is_one = true

    p = ProgressUnknown(desc="building the influence matrix", spinner=true)

    for k in 0:(n_c-2)
        B, sBA, A, sAB = superfilter_iTEBD_step(reduced_gate(n_c - k), R, R_d, A, sAB, B, sBA, tol; truncation=truncation, cap_rank=cap_rank, low_rank_svd=low_rank_svd)

        if rank_is_one
            if all([length(sAB) == 1, length(sBA) == 1])
                # reset to initial mps if rank is still one
                A, B = copy(A_init), copy(B_init)
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

    # time-local weights, with trotter corrections for even and odd time steps
    cf = η[1, :, :]
    cf2 = deepcopy(cf)
    for i in 1:m, j in 1:m
        if i != j
            cf2[i, j] = cf[i, j] + conj(cf[j, i])
        end
    end
    R_e = (1:m) .<= (1:m)'
    R_o = (1:m) .>= (1:m)'
    i_tens = zeros(Complex{ftype}, D)
    i_tens_e = zeros(Complex{ftype}, D)
    i_tens_o = zeros(Complex{ftype}, D)
    for μ in 1:D
        sl, sr = s_l[:, μ], s_r[:, μ]
        i_tens[μ] = exp(-(sl - sr)' * (cf * sl - conj.(cf) * sr))
        i_tens_e[μ] = exp((sl' * conj.(cf) * sr) + (sr' * cf * sl) - (sl' * (R_e .* cf2) * sl) - (sr' * (R_e .* conj.(cf2)) * sr))
        i_tens_o[μ] = exp((sl' * conj.(cf) * sr) + (sr' * cf * sl) - (sl' * (R_o .* cf2) * sl) - (sr' * (R_o .* conj.(cf2)) * sr))
    end

    # superoperators of the reduced legs, S[(c, s), (e, s')] = Σ_μ u_c(d(μ)) v_e(μ) i_tens(μ) P_μ[s', s], with the
    # projection P_μ ρ = F_f ρ B_g onto the eigenspaces of the forward and backward eigenvalue tuples f, g of μ (operators
    # applied in `order`); summed over f for each g, so that no superoperator per μ is formed
    tuples = vec(CartesianIndices(Tuple(r)))
    function layer_superoperator(order, weights)
        Fm = reduce(hcat, [vec(prod(projectors[l][f[l]] for l in reverse(order))) for f in tuples])
        Bm = reduce(hcat, [vec(prod(projectors[l][g[l]] for l in order)) for g in tuples])
        Y = zeros(Complex{ftype}, s_dim^2, R_d * R, length(tuples))
        for (j, g) in enumerate(tuples)
            μs = [LinearIndices(ind)[CartesianIndex(Tuple(f[l] + r[l] * (g[l] - 1) for l in 1:m))] for f in tuples]
            Y[:, :, j] = Fm * reshape(weights[μs] .* u[diff_index[μs], :] .* reshape(v[μs, :], :, 1, R), :, R_d * R)
        end
        S = reshape(reshape(Y, :, length(tuples)) * transpose(Bm), s_dim, s_dim, R_d, R, s_dim, s_dim) # [s'_l, s_l, c, e, s_r, s'_r]
        return reshape(permutedims(S, (3, 2, 5, 4, 1, 6)), R_d * s_dim^2, R * s_dim^2)
    end
    # inverting the ordering on even and odd steps to ensure symmetric trotterization (odd layer first)
    S = commuting ? [layer_superoperator(1:m, i_tens)] : [layer_superoperator(1:m, i_tens_o), layer_superoperator(m:-1:1, i_tens_e)]

    # bond tensors A[(x, c), a] and B[b, (x, e)] of the propagator, and the boundaries from A at zero difference and B
    # at the trivial index
    Aq = reshape(permutedims(A, (3, 2, 1)), :, size(A, 1))
    Bq = permutedims(reshape(reshape(sAB, :, 1, 1) .* B .* reshape(sBA, 1, 1, :), :, size(B, 3)))
    A0 = A[:, 1, :]
    B0 = dropdims(sum(reshape(v[D+1, :], 1, :, 1) .* B; dims=2); dims=2)
    @ein f0[b, a] := A0[a, x] * sAB[x] * B0[x, b] * sBA[b]
    v_l, v_r = get_boundaries_uniTEMPO(f0, ftype)

    return SparseUniformPTMPO(s_dim, delta_t_orig, SparsePropagator(Aq, Bq, S), v_r, transpose(v_l))
end
