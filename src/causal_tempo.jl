# Causal TEMPO [Richter & Cygorek, arXiv:2610.10259] for Gaussian baths with one or several coupling operators.
#
# The Trotterized influence functional is the same as in `uniTEMPO`. Instead of compressing it into a uniform process
# tensor, the path integral is contracted causally, one bath layer after the other. The state of the contraction is an
# MPS
#
#     [ρ] - [L_1] - [L_2] - ... - [L_K]
#
# The first site is the vectorized system density matrix. The "future line" L_j holds the influence of all past layers
# on the layer j steps ahead. This influence only depends on the (degeneracy filtered) eigenvalue difference
# Σ_l (s^l_i - s^l_j) of that future layer, so the lines have a small physical dimension. In every layer
#
#   1. the system is projected onto the coupling eigenspaces, which creates the path index μ of the layer and closes
#      the first line,
#   2. μ is passed through the remaining lines, line k + 1 picks up the memory weight I(k) between μ and its eigenvalue
#      difference,
#   3. μ is summed out by opening a new line at the end of the chain (the one that is n_c layers ahead),
#   4. all bonds are compressed in a sweep back to the system site.
#
# Lines of layers after the last time step are only ever evaluated at zero eigenvalue difference, where all memory
# weights are one, so they are not stored. The reduced density matrix is obtained by setting all lines to the zero
# difference, which is the trace over the future of the environment.
#
# This file is the reference implementation with the full path index μ (dimension D) and the full eigenvalue
# difference index d (dimension D_diff) of the lines. `causal_tempo_superfilter.jl` carries both in reduced polynomial
# bases, whose dimensions are set by the number of bath channels and not by the system size.


"""
    CausalInfluence

Ingredients of the Trotterized influence functional for [`causalTEMPO`](@ref), see [`causal_influence`](@ref).

# Fields
- `s_dim` system dimension.
- `delta_t` Trotter time step.
- `n_c` memory cutoff in bath layers.
- `layer_tens` one tensor for each bath layer in a time step (one for commuting couplings, two for non-commuting
  couplings). `layer_tens[l][:, μ, :]` is the superoperator that projects the system onto the eigenspaces of path
  index `μ`, times the time-local influence.
- `memory` memory weights. `memory[k][d, μ]` couples the path index `μ` of a layer to the eigenvalue difference `d`
  of the layer `k` steps later.
- `diff_index` the eigenvalue difference `d` of each path index `μ`.
- `zero_diff` the index `d` of the zero eigenvalue difference.
"""
struct CausalInfluence{T<:Number}
    s_dim::Int
    delta_t::Real
    n_c::Int
    layer_tens::Vector{Array{T,3}}
    memory::Vector{Matrix{T}}
    diff_index::Vector{Int}
    zero_diff::Int
end


"""
    causal_influence(s_ops::AbstractVector, delta_t::Real, bcf::Function, n_c::Int; ftype::DataType=Float64)

Compute the influence functional ingredients of [`causalTEMPO`](@ref) for the coupling operators `s_ops`, the bath
correlation function `bcf` (returning a matrix for several couplings) and a memory cutoff of `n_c` time steps. For
non-commuting couplings each time step is split into two bath layers of step `delta_t / 2` (second order Trotter), as
in [`uniTEMPO`](@ref).
"""
function causal_influence(s_ops::AbstractVector, delta_t::Real, bcf::Function, n_c::Int; ftype::DataType=Float64)
    lab = causal_labels(s_ops; ftype=ftype)
    s_dim, D = size(s_ops[1], 1), size(lab.s_l, 2)
    # bath layers have half the time step for non-commuting couplings (second order trotter)
    n_c = lab.commuting ? n_c : 2 * n_c
    η = causal_η(bcf, lab.commuting ? delta_t : delta_t / 2, n_c, length(s_ops), ftype)
    memory = [exp.(-transpose(lab.s_d) * (η[k+1, :, :] * lab.s_l - conj.(η[k+1, :, :]) * lab.s_r)) for k in 1:n_c]
    local_weights = causal_local_weights(η[1, :, :], lab)

    # projections ρ -> P_i ρ P_j in Trotter order: 1..m in odd layers, m..1 in even layers (symmetric second order Trotter)
    function layer_tensor(order, weights)
        tens = zeros(Complex{ftype}, s_dim^2, D, s_dim^2)
        for μ in 1:D
            sup = Matrix{Complex{ftype}}(I, s_dim^2, s_dim^2)
            for l in order
                sup = kron(transpose(lab.projectors[l][lab.bwd[l, μ]]), lab.projectors[l][lab.fwd[l, μ]]) * sup
            end
            tens[:, μ, :] = weights[μ] * sup
        end
        return tens
    end
    layer_tens = [layer_tensor(o, local_weights[:, i]) for (i, o) in enumerate(lab.orders)]

    return CausalInfluence{Complex{ftype}}(s_dim, delta_t, n_c, layer_tens, memory, lab.diff_index, lab.zero_diff)
end
function causal_influence(s::AbstractMatrix{<:Number}, delta_t::Real, bcf::Function, n_c::Int; kwargs...)
    return causal_influence([s], delta_t, t -> bcf(t) * ones(1, 1), n_c; kwargs...)
end


"""
    causal_labels(s_ops::AbstractVector; ftype::DataType=Float64)

Eigen-pair labels of the path index μ = (ν_1, ..., ν_m) of a bath layer, ν_l = i + r_l (j - 1) for the forward (i) and
backward (j) eigenvalues of coupling operator l (degeneracy filtered). Returns a named tuple with the eigenprojectors
`projectors[l][i]`, the eigenvalues `s_vals[l]`, `fwd`, `bwd` (m × D eigenvalue indices), `s_l`, `s_r` (m × D
eigenvalues), the eigenvalue differences `s_d` (m × D_diff), `diff_index` (difference of each μ), `zero_diff`,
`commuting`, and the operator `orders` of the bath layers in a time step (1..m, and m..1 for non-commuting couplings).
"""
function causal_labels(s_ops::AbstractVector; ftype::DataType=Float64)
    m = length(s_ops)
    s_dim = size(s_ops[1], 1)
    projectors = []
    s_vals = []
    filter_s_diff = []
    s_diff_red = []
    for s in s_ops
        @assert size(s) == (s_dim, s_dim) "All coupling operators must be square and of the same dimension"
        @assert s' == s "All coupling operators must be hermitian"
        s_vals_, u = eigen(Hermitian(Matrix(s)))
        p, s_vals_ = degeneracy_filter(s_vals_)
        push!(projectors, [u * Diagonal(Vector{ftype}(p[i, :])) * u' for i in axes(p, 1)])
        push!(s_vals, s_vals_)
        filter_s_diff_, s_diff_red_ = degeneracy_filter((s_vals_*ones(length(s_vals_))'.-ones(length(s_vals_))*s_vals_')[:])
        push!(filter_s_diff, filter_s_diff_)
        push!(s_diff_red, s_diff_red_)
    end
    commuting = check_commuting(s_ops; ftype=ftype)

    r = length.(s_vals)
    ind = CartesianIndices(Tuple(r .^ 2))
    D = length(ind)
    fwd = [1 + (ind[μ][l] - 1) % r[l] for l in 1:m, μ in 1:D]
    bwd = [1 + (ind[μ][l] - 1) ÷ r[l] for l in 1:m, μ in 1:D]
    s_l = [s_vals[l][fwd[l, μ]] for l in 1:m, μ in 1:D]
    s_r = [s_vals[l][bwd[l, μ]] for l in 1:m, μ in 1:D]

    # eigenvalue differences s_l - s_r (degeneracy filtered)
    ind_diff = CartesianIndices(Tuple(length.(s_diff_red)))
    s_d = [s_diff_red[l][ind_diff[d][l]] for l in 1:m, d in 1:length(ind_diff)]
    diff_index = [LinearIndices(ind_diff)[CartesianIndex(Tuple(findfirst(filter_s_diff[l][:, ind[μ][l]]) for l in 1:m))] for μ in 1:D]
    zero_diff = findfirst(d -> all(iszero, s_d[:, d]), 1:length(ind_diff))
    orders = commuting ? [1:m] : [1:m, m:-1:1]
    return (; projectors, s_vals, fwd, bwd, s_l, s_r, s_d, diff_index, zero_diff, commuting, orders)
end


"""η[k + 1, :, :] for layers k = 0, ..., n_c apart (η[1] is the time-local term), see `compute_η`."""
function causal_η(bcf::Function, layer_dt::Real, n_c::Int, m::Int, ftype::DataType)
    @assert size(bcf(0), 1) == m "Number of coupling operators and size of bath correlation functions must match."
    η = zeros(Complex{ftype}, n_c + 1, m, m)
    for i in 1:m, j in 1:m
        η[:, i, j] = Complex{ftype}.(compute_η(t -> bcf(t)[i, j], layer_dt, n_c))
    end
    return η
end


"""Time-local influence of each path index μ (as in uniTEMPO), one column per bath layer of a time step."""
function causal_local_weights(cf::AbstractMatrix, lab)
    m, D = size(lab.s_l)
    s_l, s_r = lab.s_l, lab.s_r
    local_weights = zeros(eltype(cf), D, length(lab.orders))
    if lab.commuting
        for μ in 1:D
            sl, sr = s_l[:, μ], s_r[:, μ]
            local_weights[μ, 1] = exp(-(sl - sr)' * (cf * sl - conj.(cf) * sr))
        end
    else
        cf2 = deepcopy(cf)
        for i in 1:m, j in 1:m
            if i != j
                cf2[i, j] = cf[i, j] + conj(cf[j, i])
            end
        end
        R_e = (1:m) .<= (1:m)'
        R_o = (1:m) .>= (1:m)'
        for μ in 1:D
            sl, sr = s_l[:, μ], s_r[:, μ]
            local_weights[μ, 1] = exp((sl' * conj.(cf) * sr) + (sr' * cf * sl) - (sl' * (R_o .* cf2) * sl) - (sr' * (R_o .* conj.(cf2)) * sr))
            local_weights[μ, 2] = exp((sl' * conj.(cf) * sr) + (sr' * cf * sl) - (sl' * (R_e .* cf2) * sl) - (sr' * (R_e .* conj.(cf2)) * sr))
        end
    end
    return local_weights
end


"""
    trim_memory(inf::CausalInfluence, threshold::Real)

Drop the memory beyond the last layer whose weights differ from one by at least `threshold`: lines that far ahead only
pick up weights of one within the truncation, so they are not stored.
"""
function trim_memory(inf::CausalInfluence, threshold::Real)
    k = findlast(m -> maximum(abs, m .- 1) >= threshold, inf.memory)
    n_c = isnothing(k) ? 1 : k
    return CausalInfluence(inf.s_dim, inf.delta_t, n_c, inf.layer_tens, inf.memory[1:n_c], inf.diff_index, inf.zero_diff)
end


"""Thin SVD; if LAPACK's divide-and-conquer SVD fails to converge, the slower QR iteration is used instead."""
function robust_svd(M::AbstractMatrix{<:Number})
    try
        svd(M; full=false)
    catch e
        e isa LAPACKException || rethrow()
        svd(M; full=false, alg=LinearAlgebra.QRIteration())
    end
end

"""
    causal_svd(M::AbstractMatrix{<:Number}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)

Truncated SVD as in `trunc_svd`, with two additions. Singular values at the level of round-off errors are always
discarded: all future lines exist from the start, so for `tol = 0` numerical noise would otherwise fill up the bond
dimension. If LAPACK's divide-and-conquer SVD fails to converge, the slower QR iteration is used instead.
"""
function causal_svd(M::AbstractMatrix{<:Number}, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)
    u, s, vp = robust_svd(M)
    rank_numerical = something(findlast(>(64 * eps(eltype(s)) * s[1]), s), 1)
    rank = min(new_rank(s, tol, truncation; cap_rank=cap_rank), rank_numerical)
    return u[:, 1:rank], s[1:rank], vp'[1:rank, :]
end


"""
    causal_svd_low_rank(M::AbstractMatrix{<:Number}, tol::Real, r::Int; truncation::Symbol=:rel, cap_rank::Int=100_000)

Truncated SVD through a randomized low-rank projection onto `r` dimensions, as `trunc_svd_low_rank`, but the rank is
not capped by the guess `r`: if all singular values found are still kept by the truncation, the true rank may be
larger, and the projection is repeated with twice the rank (up to the full SVD). A fixed guess silently truncates
whenever the bond grows faster than expected, independently of `tol`. `LOW_RANK_RETRIES[]` counts the retries.
"""
const LOW_RANK_RETRIES = Ref(0)
function causal_svd_low_rank(M::AbstractMatrix{<:Number}, tol::Real, r::Int; truncation::Symbol=:rel, cap_rank::Int=100_000)
    while r < min(size(M)...)
        U = try
            psvd(M; rank=max(10, r))[1]
        catch e
            e isa LAPACKException || rethrow()
            break # the randomized projection failed in LAPACK: full SVD below
        end
        u2, s, vp = robust_svd(U' * M)
        rank_numerical = something(findlast(>(64 * eps(eltype(s)) * s[1]), s), 1)
        rank = min(new_rank(s, tol, truncation; cap_rank=cap_rank), rank_numerical)
        # the spectrum is resolved if something was truncated, or if the projection found fewer than r directions
        if rank < length(s) || length(s) < max(10, r) || rank >= cap_rank
            return (U*u2)[:, 1:rank], s[1:rank], vp'[1:rank, :]
        end
        LOW_RANK_RETRIES[] += 1
        r *= 2
    end
    return causal_svd(M, tol; truncation=truncation, cap_rank=cap_rank)
end


"""
    causal_layer!(mps::Vector, inf::CausalInfluence, layer::Int, n_lines::Int, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)

Contract one bath layer into `mps = [ρ, L_1, ..., L_K]`, which has its orthogonality center on the system site.
Afterwards the chain holds `n_lines` future lines: `K`, or `K - 1` if the end of the simulation is closer than the
memory cutoff. `layer` selects the Trotter layer within the time step. Returns the log of the norm that is divided out.
"""
function causal_layer!(mps::Vector, inf::CausalInfluence, layer::Int, n_lines::Int, tol::Real; truncation::Symbol=:rel, cap_rank::Int=100_000)
    svd_t(M) = causal_svd(M, tol; truncation=truncation, cap_rank=cap_rank)
    tens = inf.layer_tens[layer]
    d2, D, _ = size(tens)
    D_diff = size(inf.memory[1], 1)
    K = length(mps) - 1
    @assert n_lines == K || n_lines == K - 1

    # 1. project the system and close the first line with the eigenvalue difference of μ: θ[s_out, μ, z]
    ρ = reshape(mps[1], d2, :)
    @tensor ρ_μ[s_out, μ, y] := tens[s_out, μ, s_in] * ρ[s_in, y]
    line = mps[2]
    θ = zeros(eltype(ρ_μ), d2, D, size(line, 3))
    for μ in 1:D
        θ[:, μ, :] = ρ_μ[:, μ, :] * line[:, inf.diff_index[μ], :]
    end
    u, s, v = svd_t(reshape(θ, d2, :))
    new = [reshape(u, 1, d2, :)]
    # the norm carried to the right is divided out on every line: it grows by about √D_diff per line and overflows
    # Float64 after a few hundred lines
    log_norm = log(norm(s))
    M = reshape(Diagonal(s ./ norm(s)) * v, :, D, size(line, 3)) # carries μ to the right

    # 2. pass μ through the remaining lines, line k + 1 is k layers ahead: θ[x, d, μ, z]
    for k in 1:K-1
        line = mps[k+2]
        @tensor θ[x, d, μ, z] := M[x, μ, y] * line[y, d, z]
        θ .*= reshape(inf.memory[k], 1, D_diff, D, 1)
        x, _, _, z = size(θ)
        # the bond to the right of this line, from the previous sweep, sets the rank to look for
        u, s, v = causal_svd_low_rank(reshape(θ, x * D_diff, D * z), tol, 2 * size(line, 1) + 16; truncation=truncation, cap_rank=cap_rank)
        push!(new, reshape(u, x, D_diff, :))
        log_norm += log(norm(s))
        M = reshape(Diagonal(s ./ norm(s)) * v, :, D, z)
    end

    # 3. sum out μ by opening the line that is K layers ahead (unless it lies beyond the end of the simulation)
    M = reshape(M, size(M, 1), D) # the last line has a trivial right bond
    if n_lines == K
        push!(new, reshape(M * transpose(inf.memory[K]), :, D_diff, 1))
    else
        new[end] = reshape(reshape(new[end], :, size(M, 1)) * sum(M, dims=2), size(new[end], 1), size(new[end], 2), 1)
    end

    log_norm += compress_backwards!(new, svd_t)
    resize!(mps, length(new))
    mps .= new
    return log_norm
end


"""
    compress_backwards!(mps::Vector, svd_t::Function)

Truncated SVD sweep from the last site to the first one, which moves the orthogonality center to the first site. The
first site is normalized and the log of its norm is returned.
"""
function compress_backwards!(mps::Vector, svd_t::Function)
    for i in length(mps):-1:2
        a, p, z = size(mps[i])
        u, s, v = svd_t(reshape(mps[i], a, p * z))
        mps[i] = reshape(v, :, p, z)
        a0, p0, _ = size(mps[i-1])
        mps[i-1] = reshape(reshape(mps[i-1], a0 * p0, a) * u * Diagonal(s), a0, p0, :)
    end
    nrm = norm(mps[1])
    mps[1] ./= nrm
    return log(nrm)
end


"""
    causal_readout(mps::Vector, inf::CausalInfluence)

Reduced density matrix of the system (up to the stored normalization), obtained by setting all future lines to zero
eigenvalue difference.
"""
function causal_readout(mps::Vector, inf::CausalInfluence)
    ρ, log_scale = causal_readout_log(mps, inf)
    return exp(log_scale) * ρ
end

# readout as (ρ / e^log_scale, log_scale): the product over hundreds of lines under- or overflows otherwise
function causal_readout_log(mps::Vector, inf::CausalInfluence)
    R = ones(eltype(mps[1]), 1, 1)
    log_scale = 0.0
    for i in length(mps):-1:2
        R = mps[i][:, inf.zero_diff, :] * R
        nrm = norm(R)
        R ./= nrm
        log_scale += log(nrm)
    end
    return reshape(mps[1][1, :, :] * R, inf.s_dim, inf.s_dim), log_scale
end


"""
    causalTEMPO(s::Union{AbstractMatrix{<:Number}, AbstractVector}, delta_t::Real, bcf::Function, tol::Real, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s=nothing, n_c::Union{Int,Symbol}=:auto, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, return_bond_dims::Bool=false, callback=nothing)

Compute the time evolution of the reduced system for `n` time steps with the causal TEMPO algorithm, starting from the
product state of the system state `ρ_0` and the bath in equilibrium. Returns a vector of density matrices at each time
step, as [`evolve`](@ref).

In contrast to [`uniTEMPO`](@ref), causal TEMPO does not compute a process tensor but contracts the path integral for
the given dynamics directly, step by step [Richter & Cygorek, arXiv:2610.10259]. It takes the full memory into account
by default and remains efficient at strong coupling and for long-lived memory, where the process tensor of uniTEMPO
needs very large bond dimensions. The cost per time step grows with the number of memory lines, i.e. linearly in time
up to the memory cutoff. This function carries the full path index and eigenvalue difference index, whose dimensions
grow exponentially with the number of coupling operators; for several couplings use [`causalTEMPO_superfilter`](@ref),
whose bulk dimensions are set by the number of bath channels only.

# Arguments
- `s` is the coupling operator (must be hermitian) or a list of coupling operators.
- `delta_t` is the Trotter time step.
- `bcf` is the bath correlation function (returning a matrix for several coupling operators, as in `uniTEMPO`).
- `tol` is the SVD compression tolerance, determining the accuracy of the compression.
- `ρ_0` is the initial state of the system.
- `n` is the number of time steps.

# Keyword Arguments
- `h_s` is the system Hamiltonian, may also be time dependent (a function of time).
- `n_c` is the memory cutoff in time steps. By default (`:auto`) the memory ends where the memory weights differ from
  one by less than `tol`.
- `truncation` specifies the truncation scheme for singular value truncation. It can be `:rel` for relative, `:abs` for absolute, see function `new_rank`.
- `cap_rank` fixes the maximum bond dimension allowed during the SVD compression. Usually it is not recommended to use this option.
- `return_bond_dims` if set to true, the maximal bond dimension after each time step is returned as a second output.
- `callback` is called as `callback(i, ρ, χ)` after time step `i` with the density matrix and the maximal bond
  dimension, e.g. to write results while a long run is going. The state at step `i` does not change afterwards.

# Example
```
bcf(t) = 0.5 * exp(-abs(t) - im * t)
ρ_t = causalTEMPO(sz, 0.1, bcf, 1e-7, [1 0; 0 0], 100; h_s=sx)
plot(real.([tr(sz * ρ) for ρ in ρ_t])) # plot the dynamics of ⟨σ_z⟩
```
"""
function causalTEMPO(s::Union{AbstractMatrix{<:Number},AbstractVector}, delta_t::Real, bcf::Function, tol::Real, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing, n_c::Union{Int,Symbol}=:auto, truncation::Symbol=:rel, cap_rank::Int=100_000, ftype::DataType=Float64, return_bond_dims::Bool=false, callback=nothing)
    inf = causal_influence(s, delta_t, bcf, n_c == :auto ? n : min(n_c, n); ftype=ftype)
    n_c == :auto && (inf = trim_memory(inf, tol))
    return causalTEMPO(inf, tol, ρ_0, n; h_s=h_s, truncation=truncation, cap_rank=cap_rank, return_bond_dims=return_bond_dims, callback=callback)
end
function causalTEMPO(inf::CausalInfluence, tol::Real, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing, truncation::Symbol=:rel, cap_rank::Int=100_000, return_bond_dims::Bool=false, callback=nothing)
    @assert size(ρ_0) == (inf.s_dim, inf.s_dim) "Initial state has invalid shape."
    T = eltype(inf.layer_tens[1])
    d2 = inf.s_dim^2
    dt = inf.delta_t
    n_layers = length(inf.layer_tens) # bath layers per time step

    # system evolution for half a time step before and after the bath layers of each time step
    if isnothing(h_s)
        h_s = zeros(T, inf.s_dim, inf.s_dim)
    end
    if h_s isa Function
        half_step = t -> T.(local_channel((t, t + dt / 2), h_s))
    else
        @assert size(h_s) == (inf.s_dim, inf.s_dim) "Hamiltonian has invalid shape."
        u = T.(kron(transpose(exp(im * h_s * dt / 2)), exp(-im * h_s * dt / 2)))
        half_step = t -> u
    end
    evolve_system!(mps, t) = (mps[1] = reshape(half_step(t) * reshape(mps[1], d2, :), 1, d2, :))

    # all future lines are trivial for the product initial state, lines after the last time step are not stored
    remaining_layers = n_layers * n
    D_diff = size(inf.memory[1], 1)
    mps = [reshape(T.(ρ_0[:]), 1, d2, 1)]
    append!(mps, [ones(T, 1, D_diff, 1) for _ in 1:min(inf.n_c, remaining_layers)])
    log_norm = 0.0

    ρ_t = Vector{Matrix{T}}(undef, n + 1)
    ρ_t[1] = T.(ρ_0)
    bond_dims = zeros(Int, n)

    @showprogress showspeed = true desc = "performing causal TEMPO..." for i in 1:n
        t = (i - 1) * dt
        evolve_system!(mps, t)
        for layer in 1:n_layers
            remaining_layers -= 1
            log_norm += causal_layer!(mps, inf, layer, min(inf.n_c, remaining_layers), tol; truncation=truncation, cap_rank=cap_rank)
        end
        evolve_system!(mps, t + dt / 2)
        ρ_r, log_scale = causal_readout_log(mps, inf)
        ρ_t[i+1] = exp(log_norm + log_scale) * ρ_r
        bond_dims[i] = maximum(size(a, 3) for a in mps)
        isnothing(callback) || callback(i, ρ_t[i+1], bond_dims[i])
    end

    return_bond_dims && (return ρ_t, bond_dims)
    return ρ_t
end
