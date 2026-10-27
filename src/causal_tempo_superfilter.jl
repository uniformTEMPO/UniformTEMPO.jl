# Causal TEMPO with both legs of the bath layers reduced to polynomials in the bath-channel coordinates of the
# eigenvalue labels ("super filtering").
#
# A memory weight between the path index μ of a layer and the eigenvalue difference d of the layer k steps later is
#
#     exp(-Δ(d)ᵀ (η_k s^l(μ) - η_k* s^r(μ))) = exp(w(d)ᵀ T_k ζ(μ)),
#
# with the path coordinates ζ = (Q† s^l, Qᵀ s^r) (Q spans the row space of all η_k, one column per bath channel) and the
# line coordinates w = Pᵀ Δ (P spans the column spaces of all η_k Q and their conjugates). The exponent is O(Δt²). The
# carried path index (after the system site) and the future lines are kept in orthonormal bases of the polynomials of
# degree ≤ p in ζ and in w (the lines with the zero difference kept exact, `mu_degree` and `line_degree`), and the
# weights are applied in their Taylor form of order `taylor_order`: the reduced gates are
# combinations of the projected monomials of ζ and w, so the D_diff × D memory weights are never formed. The system
# site projects onto the reduced path basis directly, with the (precomputed) boundary tensor
#
#     B[s_out, a, c, s_in] = Σ_μ Φ[μ, a] U[d(μ), c] w_loc(μ) (P_μ)[s_out, s_in],
#
# so that no index of the size of the full path index D enters the layer contraction. The dimensions of the bulk are set
# by the number of bath channels r and not by the system: R = (r + 1)(2r + 1) for p = 2.


"""Exponent vectors of the monomials in `nv` variables up to degree `M` (ordered by degree), their degrees, and
`next[i, a]`, the index of `E[i] + e_a` (0 above degree `M`)."""
function monomials(nv::Int, M::Int)
    E = [zeros(Int, nv)]
    for q in 1:M, e in [e for e in E if sum(e) == q - 1], a in 1:nv
        f = copy(e)
        f[a] += 1
        f in E || push!(E, f)
    end
    idx = Dict(e => i for (i, e) in enumerate(E))
    return E, sum.(E), [get(idx, setindex!(copy(e), e[a] + 1, a), 0) for e in E, a in 1:nv]
end


"""Coefficients `C[i, j]` of the monomials w^EW[i] ζ^EZ[j] in Σ_{q ≤ M} (wᵀ T ζ)^q / q!, for the monomials up to degree
`M` (degrees `dW`, `dZ` and successor indices `nextW`, `nextZ` from `monomials`)."""
function taylor_coefficients(T::AbstractMatrix, M::Int, dW, nextW, dZ, nextZ)
    nW, nZ = count(<=(M), dW), count(<=(M), dZ)
    C = zeros(eltype(T), nW, nZ)
    C[1, 1] = 1
    Cq = copy(C)
    for q in 1:M
        Cn = zeros(eltype(T), nW, nZ)
        for i in findall(==(q - 1), dW), j in findall(==(q - 1), dZ), a in axes(T, 1), b in axes(T, 2)
            Cn[nextW[i, a], nextZ[j, b]] += Cq[i, j] * T[a, b] / q
        end
        C .+= (Cq = Cn)
    end
    return C
end


"""Orthonormal basis of the column span of `A` (singular values above `tol` relative to the largest)."""
orthonormal_span(A::AbstractMatrix, tol::Real) = (F = svd(A); F.U[:, 1:count(>(tol * F.S[1]), F.S)])


"""
    memory_cutoff(η, lab, tol)

Last layer distance k at which a memory weight exp(-Δᵀ(η_k s^l - η_k* s^r)) differs from one by at least `tol` (the
criterion of [`trim_memory`](@ref)), without forming the weights: the exponent is a[d, f] + b[d, g] for the forward
and backward eigenvalue tuples f, g, and exp(max|a| + max|b|) - 1 < tol rules out a distance at the cost of a and b.
"""
function memory_cutoff(η::AbstractArray{<:Number,3}, lab, tol::Real)
    m = size(η, 2)
    s_t = [lab.s_vals[l][t[l]] for l in 1:m, t in vec(CartesianIndices(Tuple(length.(lab.s_vals))))]
    for k in size(η, 1)-1:-1:1
        a = -transpose(lab.s_d) * η[k+1, :, :] * s_t
        b = transpose(lab.s_d) * conj.(η[k+1, :, :]) * s_t
        exp(maximum(maximum(abs, a; dims=2) .+ maximum(abs, b; dims=2))) - 1 < tol && continue
        for d in axes(a, 1), i in axes(a, 2), j in axes(b, 2)
            abs(exp(a[d, i] + b[d, j]) - 1) >= tol && return k
        end
    end
    return 1
end


"""
    SuperfilterInfluence

Influence functional of [`causalTEMPO_superfilter`](@ref) in the reduced bases Φ (path index, D × R) and U (lines,
D_diff × R_d, first column the zero difference), see [`superfilter_influence`](@ref).

# Fields
- `s_dim` system dimension.
- `delta_t` Trotter time step.
- `n_c` memory cutoff in bath layers.
- `boundary` one matrix per bath layer of a time step, `boundary[l][(s_out, a), (c, s_in)]` = B: projection of the
  system onto the eigenspaces of μ times the time-local influence, paired with Φ[μ, a] and the first line at d(μ),
  U[d(μ), c].
- `Pz` projected path monomials, `Pz[j, (a, b)] = (Φ† diag(ζ^EZ[j]) Φ)[a, b]`.
- `Pw` projected line monomials, `Pw[i, (c', c)] = (U† diag(w^EW[i]) U)[c', c]`.
- `coef` Taylor coefficients of the memory weights, `coef[k][i, j]` of w^EW[i] ζ^EZ[j] for layer distance k.
- `closing` the weights of the line that closes the chain, `closing[a, c'] = Σ_{μ, d} Φ*[μ, a] I_{n_c}(d, μ) U*[d, c']`.
- `c_one` `Φ† 1`: sum over μ.
- `line_one` `U† 1`: a trivial line.
- `gates` the reduced gates of all layer distances (see [`superfilter_gate`](@ref)) if they fit into the memory
  budget, otherwise empty and generated in every layer.
"""
struct SuperfilterInfluence{T<:Number}
    s_dim::Int
    delta_t::Real
    n_c::Int
    boundary::Vector{Matrix{T}}
    Pz::Matrix{T}
    Pw::Matrix{T}
    coef::Vector{Matrix{T}}
    closing::Matrix{T}
    c_one::Vector{T}
    line_one::Vector{T}
    gates::Vector{Matrix{T}}
    ranks::Vector{Int} # bond rank found at each line in the last layer: the starting rank of the randomized SVD in the next (0 = none yet)
end


"""Reduced gate of layer distance k as the matrix `G[(a, c), (c', b)]`: path coefficient a → b, line coefficient c → c'."""
function superfilter_gate(sf::SuperfilterInfluence, k::Int)
    isempty(sf.gates) || return sf.gates[k]
    return taylor_gate(sf.Pw, sf.coef[k], sf.Pz)
end
function taylor_gate(Pw::AbstractMatrix, C::AbstractMatrix, Pz::AbstractMatrix)
    R, Rd = isqrt(size(Pz, 2)), isqrt(size(Pw, 2))
    return reshape(permutedims(reshape(transpose(Pw) * C * Pz, Rd, Rd, R, R), (3, 2, 1, 4)), R * Rd, Rd * R)
end


"""
    superfilter_boundary(lab, Φ, U, weights, order)

Boundary matrix `B[(s_out, a), (c, s_in)] = Σ_μ Φ[μ, a] U[d(μ), c] weights[μ] (P_μ)[s_out, s_in]` of a bath layer with
the coupling operators applied in `order`. The projection is P_μ ρ = F_f ρ B_g, with ordered products of the
eigenprojectors for the forward and backward eigenvalue tuples f, g of μ, and the sum over f is done first for each g,
so that only the D_f + D_g products and no superoperator per μ are formed.
"""
function superfilter_boundary(lab, Φ::AbstractMatrix, U::AbstractMatrix, weights::AbstractVector, order)
    n = size(lab.projectors[1][1], 1)
    r = length.(lab.s_vals)
    R, Rd = size(Φ, 2), size(U, 2)
    tuples = vec(CartesianIndices(Tuple(r)))
    μ_index = LinearIndices(Tuple(r .^ 2))
    Fm = reduce(hcat, [vec(prod(lab.projectors[l][f[l]] for l in reverse(order))) for f in tuples]) # vec(F_f)
    Bm = reduce(hcat, [vec(prod(lab.projectors[l][g[l]] for l in order)) for g in tuples])          # vec(B_g)
    Y = zeros(eltype(Φ), n^2, R * Rd, length(tuples))
    for (j, g) in enumerate(tuples)
        idx = [μ_index[CartesianIndex(Tuple(f[l] + r[l] * (g[l] - 1) for l in eachindex(r)))] for f in tuples]
        κ = weights[idx] .* Φ[idx, :] .* reshape(U[lab.diff_index[idx], :], :, 1, Rd)
        Y[:, :, j] = Fm * reshape(κ, :, R * Rd)
    end
    B = reshape(reshape(Y, :, length(tuples)) * transpose(Bm), n, n, R, Rd, n, n) # [ro, ri, a, c, ci, co]
    return reshape(permutedims(B, (1, 6, 3, 4, 2, 5)), n^2 * R, Rd * n^2)
end


"""
    superfilter_influence(s_ops::AbstractVector, delta_t::Real, bcf::Function, n_c::Int; trim::Real=0, mu_degree::Int=2, line_degree::Int=2, taylor_order::Int=2, basis_tol::Real=1e-12, ftype::DataType=Float64)

Influence functional of [`causalTEMPO_superfilter`](@ref) for the coupling operators `s_ops`, the bath correlation
function `bcf` and a memory cutoff of `n_c` time steps (bath layers of step `delta_t / 2` for non-commuting couplings,
as in [`causal_influence`](@ref)). With `trim > 0` the memory is cut after the last layer whose weights differ from one
by at least `trim`. The bases are the polynomials of degree ≤ `mu_degree` in the path coordinates and of degree ≤
`line_degree` in the line coordinates (plus the zero difference), orthonormalized with singular values above
`basis_tol`; the gates are the Taylor polynomials of order `taylor_order` of the memory weights, stored if they take
at most `gate_memory` bytes (otherwise generated in every layer).
"""
function superfilter_influence(s_ops::AbstractVector, delta_t::Real, bcf::Function, n_c::Int; trim::Real=0, mu_degree::Int=2, line_degree::Int=2, taylor_order::Int=2, basis_tol::Real=1e-12, gate_memory::Real=4e9, ftype::DataType=Float64)
    lab = causal_labels(s_ops; ftype=ftype)
    n_c = lab.commuting ? n_c : 2 * n_c
    η = causal_η(bcf, lab.commuting ? delta_t : delta_t / 2, n_c, length(s_ops), ftype)
    trim > 0 && (n_c = memory_cutoff(η, lab, trim))
    ηk = [η[k+1, :, :] for k in 1:n_c]

    # channel coordinates: η_k = X_k Q†, exponent wᵀ T_k ζ, ζ and w scaled to at most one
    F = svd(reduce(vcat, ηk))
    Q = F.V[:, 1:count(>(1e-10 * F.S[1]), F.S)]
    X = [e * Q for e in ηk]
    Fp = svd(reduce(hcat, [hcat(x, conj.(x)) for x in X]))
    P = Fp.U[:, 1:count(>(1e-10 * Fp.S[1]), Fp.S)]
    ζ = hcat(transpose(lab.s_l) * conj.(Q), transpose(lab.s_r) * Q) # D × 2r
    w = transpose(lab.s_d) * P                                     # D_diff × r_w
    zmax, wmax = maximum(abs, ζ), maximum(abs, w)
    ζ, w = ζ ./ zmax, w ./ wmax
    T = [hcat(-P' * x, P' * conj.(x)) .* (zmax * wmax) for x in X]

    EZ, dZ, nextZ = monomials(size(ζ, 2), max(taylor_order, mu_degree))
    EW, dW, nextW = monomials(size(w, 2), max(taylor_order, line_degree))
    mono = [prod(ζ[μ, :] .^ e) for μ in axes(ζ, 1), e in EZ]
    wmono = [prod(w[d, :] .^ e) for d in axes(w, 1), e in EW]

    # bases: polynomials of degree ≤ p; for the lines with the zero difference d₀ split off as the first column
    Φ = orthonormal_span(mono[:, dZ.<=mu_degree], basis_tol)
    Qd = wmono[:, dW.<=line_degree]
    Qd[lab.zero_diff, :] .= 0
    e0 = zeros(eltype(Qd), size(Qd, 1))
    e0[lab.zero_diff] = 1
    U = hcat(e0, orthonormal_span(Qd, basis_tol))

    nz, nw = count(<=(taylor_order), dZ), count(<=(taylor_order), dW)
    Pz = permutedims(reduce(hcat, [vec(Φ' * (mono[:, j] .* Φ)) for j in 1:nz]))
    Pw = permutedims(reduce(hcat, [vec(U' * (wmono[:, i] .* U)) for i in 1:nw]))
    coef = [taylor_coefficients(t, taylor_order, dW, nextW, dZ, nextZ) for t in T]
    closing = (Φ' * mono[:, 1:nz]) * transpose(coef[n_c]) * transpose(U' * wmono[:, 1:nw])

    local_weights = causal_local_weights(η[1, :, :], lab)
    boundary = [superfilter_boundary(lab, Φ, U, local_weights[:, i], o) for (i, o) in enumerate(lab.orders)]
    T_ = eltype(Φ)
    store = n_c * (size(Φ, 2) * size(U, 2))^2 * sizeof(T_) <= gate_memory
    gates = store ? [taylor_gate(Pw, C, Pz) for C in coef] : Matrix{T_}[]
    return SuperfilterInfluence{T_}(size(s_ops[1], 1), delta_t, n_c, boundary, Pz, Pw, coef, closing, Φ' * ones(T_, size(Φ, 1)), U' * ones(T_, size(U, 1)), gates, zeros(Int, n_c + 1))
end
function superfilter_influence(s::AbstractMatrix{<:Number}, delta_t::Real, bcf::Function, n_c::Int; kwargs...)
    return superfilter_influence([s], delta_t, t -> bcf(t) * ones(1, 1), n_c; kwargs...)
end


"""
    superfilter_layer!(mps::Vector, sf::SuperfilterInfluence, layer::Int, n_lines::Int, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)

Contract one bath layer into `mps = [ρ, L_1, ..., L_K]` (lines in the reduced basis), as [`causal_layer!`](@ref).
Returns the log of the norm that is divided out.
"""
function superfilter_layer!(mps::Vector, sf::SuperfilterInfluence, layer::Int, n_lines::Int, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)
    svd_t(M) = causal_svd(M, tol; truncation=truncation, cap_rank=cap_rank)
    B = sf.boundary[layer]
    R, Rd = length(sf.c_one), length(sf.line_one)
    d2 = sf.s_dim^2
    K = length(mps) - 1
    @assert n_lines == K || n_lines == K - 1

    # 1. project the system onto the path basis and close the first line: θ[s_out, a, z]
    ρ = reshape(mps[1], d2, :)
    line = mps[2]
    @tensor X[c, s_in, z] := line[y, c, z] * ρ[s_in, y]
    θ = B * reshape(X, Rd * d2, :)
    u, s, v = svd_t(reshape(θ, d2, :))
    new = [reshape(u, 1, d2, :)]
    log_norm = log(norm(s))
    M = reshape(Diagonal(s ./ norm(s)) * v, :, R, size(line, 3))

    # 2. pass the path coefficients through the remaining lines, line k + 1 is k layers ahead: θk[x, (c', b), z]
    for k in 1:K-1
        line = mps[k+2]
        Gk = superfilter_gate(sf, k)
        x, z = size(M, 1), size(line, 3)
        Tk = reshape(reshape(M, x * R, :) * reshape(line, size(line, 1), :), x, R * Rd, z) # Tk[x, (a, c), z]
        if x < 24 # small bonds: one GEMM over all (x, z), the GEMMs per z are inefficient
            θk = permutedims(reshape(reshape(permutedims(Tk, (1, 3, 2)), x * z, :) * Gk, x, z, :), (1, 3, 2))
        else      # large bonds: one GEMM per z, without permutations
            θk = similar(Tk, x, Rd * R, z)
            for j in 1:z
                mul!(view(θk, :, :, j), view(Tk, :, :, j), Gk)
            end
        end
        # rank guess: the rank of the next line in the previous layer (the lines move one site towards the system per
        # layer), with a margin so that the randomized projection resolves the spectrum without a retry
        r_prev = k + 1 <= length(sf.ranks) ? sf.ranks[k+1] : 0
        guess = r_prev > 0 ? r_prev + 10 + r_prev ÷ 8 : 2 * size(line, 1) + 16
        u, s, v = causal_svd_low_rank(reshape(θk, x * Rd, R * z), tol, guess; truncation=truncation, cap_rank=cap_rank)
        k <= length(sf.ranks) && (sf.ranks[k] = length(s))
        push!(new, reshape(u, x, Rd, :))
        log_norm += log(norm(s))
        M = reshape(Diagonal(s ./ norm(s)) * v, :, R, z)
    end

    # 3. sum out the path index by opening the line that is K layers ahead (unless it lies beyond the end)
    M = reshape(M, size(M, 1), R)
    if n_lines == K
        push!(new, reshape(M * sf.closing, :, Rd, 1))
    else
        new[end] = reshape(reshape(new[end], :, size(M, 1)) * (M * sf.c_one), size(new[end], 1), size(new[end], 2), 1)
    end

    log_norm += compress_backwards!(new, svd_t)
    resize!(mps, length(new))
    mps .= new
    return log_norm
end


# readout as (ρ / e^log_scale, log_scale): all lines at zero difference, the first function of the line basis
function superfilter_readout_log(mps::Vector, s_dim::Int)
    R = ones(eltype(mps[1]), 1, 1)
    log_scale = 0.0
    for i in length(mps):-1:2
        R = mps[i][:, 1, :] * R
        nrm = norm(R)
        R ./= nrm
        log_scale += log(nrm)
    end
    return reshape(mps[1][1, :, :] * R, s_dim, s_dim), log_scale
end


"""
    causalTEMPO_superfilter(s::Union{AbstractMatrix{<:Number}, AbstractVector}, delta_t::Real, bcf::Function, tol::Real, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s=nothing, n_c::Union{Int,Symbol}=:auto, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, return_bond_dims::Bool=false, callback=nothing, mu_degree::Int=2, line_degree::Int=2, taylor_order::Int=2)

Causal TEMPO as [`causalTEMPO`](@ref), with the path index and the future lines in bases of polynomials in the
bath-channel coordinates of the eigenvalue labels and the memory weights in Taylor form (see
[`superfilter_influence`](@ref)). Neither the memory weights nor any index of the size of the path index D or of the
eigenvalue differences D_diff enter the layer contraction, whose dimensions are set by the number of bath channels
(R = 28, R_d = 29 for three RWA emitters in a common bath, D = 4096, D_diff = 729). The error of the reduction is
O(Δt²) for `mu_degree = line_degree = taylor_order = 2`, the order of the Trotter splitting.

The arguments and the other keyword arguments are those of [`causalTEMPO`](@ref); `n_c = :auto` cuts the memory where
all weights are one within `tol`, without forming them.

# Example
```
bcf(t) = 0.5 * exp(-abs(t) - im * t)
ρ_t = causalTEMPO_superfilter(sz, 0.1, bcf, 1e-7, [1 0; 0 0], 100; h_s=sx)
```
"""
function causalTEMPO_superfilter(s::Union{AbstractMatrix{<:Number},AbstractVector}, delta_t::Real, bcf::Function, tol::Real, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing, n_c::Union{Int,Symbol}=:auto, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, return_bond_dims::Bool=false, callback=nothing, mu_degree::Int=2, line_degree::Int=2, taylor_order::Int=2)
    sf = superfilter_influence(s, delta_t, bcf, n_c == :auto ? n : min(n_c, n); trim=(n_c == :auto ? tol : 0), mu_degree=mu_degree, line_degree=line_degree, taylor_order=taylor_order, ftype=ftype)
    return causalTEMPO_superfilter(sf, tol, ρ_0, n; h_s=h_s, truncation=truncation, cap_rank=cap_rank, return_bond_dims=return_bond_dims, callback=callback)
end
function causalTEMPO_superfilter(sf::SuperfilterInfluence, tol::Real, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing, truncation::Symbol=:rel, cap_rank::Int=100_000, return_bond_dims::Bool=false, callback=nothing)
    @assert size(ρ_0) == (sf.s_dim, sf.s_dim) "Initial state has invalid shape."
    T = eltype(sf.closing)
    d2 = sf.s_dim^2
    dt = sf.delta_t
    n_layers = length(sf.boundary) # bath layers per time step

    # system evolution for half a time step before and after the bath layers of each time step
    if isnothing(h_s)
        h_s = zeros(T, sf.s_dim, sf.s_dim)
    end
    if h_s isa Function
        half_step = t -> T.(local_channel((t, t + dt / 2), h_s))
    else
        @assert size(h_s) == (sf.s_dim, sf.s_dim) "Hamiltonian has invalid shape."
        u = T.(kron(transpose(exp(im * h_s * dt / 2)), exp(-im * h_s * dt / 2)))
        half_step = t -> u
    end
    evolve_system!(mps, t) = (mps[1] = reshape(half_step(t) * reshape(mps[1], d2, :), 1, d2, :))

    # all future lines are trivial for the product initial state, lines after the last time step are not stored
    remaining_layers = n_layers * n
    mps = [reshape(T.(ρ_0[:]), 1, d2, 1)]
    append!(mps, [reshape(copy(sf.line_one), 1, :, 1) for _ in 1:min(sf.n_c, remaining_layers)])
    log_norm = 0.0

    ρ_t = Vector{Matrix{T}}(undef, n + 1)
    ρ_t[1] = T.(ρ_0)
    bond_dims = zeros(Int, n)

    @showprogress showspeed = true desc = "performing causal TEMPO (super filtered)..." for i in 1:n
        t = (i - 1) * dt
        evolve_system!(mps, t)
        for layer in 1:n_layers
            remaining_layers -= 1
            log_norm += superfilter_layer!(mps, sf, layer, min(sf.n_c, remaining_layers), tol; truncation=truncation, cap_rank=cap_rank)
        end
        evolve_system!(mps, t + dt / 2)
        ρ_r, log_scale = superfilter_readout_log(mps, sf.s_dim)
        ρ_t[i+1] = exp(log_norm + log_scale) * ρ_r
        bond_dims[i] = maximum(size(a, 3) for a in mps)
        isnothing(callback) || callback(i, ρ_t[i+1], bond_dims[i])
    end

    return_bond_dims && (return ρ_t, bond_dims)
    return ρ_t
end
