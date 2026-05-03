# PTMPO struct and evolution functions.

import Base.+
import Base.*
import Base: ^
import Base.eltype
include("uniform_MPS.jl")

"""
    UnfiromPTMPO(s_dim::Int, delta_t::Real, q::AbstractArray{<:Number,4}, v_r::AbstractVector{<:Number}, v_l::Transpose{<:Number,<:AbstractVector})
    UniformPTMPO(s_dim::Int, delta_t::Real)

a struct representing a Uniform process tensor MPO. The second constructor generates a trivial process tensor. 

# Fields
- `s_dim` system dimension
- `delta_t` Trotter time-step
- `q` uniform tensor of the UniformPTMPO (system-bath propagator). Dimensions are (aux-out, sys-out, aux-in, sys-in)
- `v_r` right boundary vector (bath initial state)
- `v_l` left boundary vector (bath trace). Must be transpose type.

# Related Functions
- `bond_dim` to return the bond dimension.
- `+` for adding two process tensors.
- `*` for combining a process tensor with a local channel.
- `^` for taking powers (changing the time-step). 
- `steadystate` to compute the steady state.
- `evolve` for density matrix time evolution.
- `channel` and `choi_channel` to compute quantum channels.
- `process_tensor` to compute process tensors.
- `two_point_correlations`, `two_point_correlations_freq` and `susceptibility` to compute unequal time correlation functions.
- `spectrum` to compute the transfer matrix spectrum.
- `truncate` to truncate the bond dimension of the process tensor.
"""
struct UniformPTMPO
    s_dim::Int
    delta_t::Real
    q::AbstractArray{<:Number,4}
    v_r::AbstractVector{<:Number}
    v_l::Transpose{<:Number,<:AbstractVector}
end
function UniformPTMPO(s_dim::Int, delta_t::Real)
    @assert delta_t > 0 "Time step delta_t should be positive."
    q = Array{ComplexF64,4}(reshape(Matrix(I, s_dim^2, s_dim^2), 1, s_dim^2, 1, s_dim^2))
    v_r = ones(ComplexF64, 1)
    v_l = ones(ComplexF64, 1)
    return UniformPTMPO(s_dim, delta_t, q, v_r, transpose(v_l))
end
function (pt::UniformPTMPO)(μ_path::AbstractVector{<:Int})
    val = pt.v_r
    for t in eachindex(μ_path)
        val = pt.q[:, μ_path[t], :, μ_path[t]] * val
    end
    return pt.v_l * val
end


"""
    to_ITensor(pt::UniformPTMPO; tags=[])

Convert a UniformPTMPO to a UniformPTMPOITensor where the data is stored as ITensors. Load ITensors.jl via `using ITensors` for this function to work.
"""
function to_ITensor end


"""
    include_system_hamiltonian(pt::UniformPTMPO, h_s::AbstractMatrix{<:Number})

Include an additional system Hamiltonian `h_s` into the process tensor `pt` via Trotterization.
"""
function include_system_hamiltonian(pt::UniformPTMPO, h_s::Union{AbstractMatrix{<:Number},Nothing})
    if isnothing(h_s)
        return pt
    end
    @assert size(h_s) == (pt.s_dim, pt.s_dim) "Hamiltonian has invalid shape."
    u = kron(transpose(exp(im * h_s * pt.delta_t / 2)), (exp(-im * h_s * pt.delta_t / 2)))
    return u * pt * u
end


"""
    bond_dim(pt::UniformPTMPO)

Returns the bond dimension of a process tensor.
"""
function bond_dim(pt::UniformPTMPO)
    return size(pt.q, 1)
end


function eltype(pt::UniformPTMPO)
    return eltype(pt.q)
end


"""
    +(a::UniformPTMPO, b::UniformPTMPO)

combine two Uniform PT-MPOs `a` and `b` in second order Trotter using the ACE combination scheme [Cygorek et.al., Nature Physics 18, 6 662-668 (2022)]. 
"""
function +(a::UniformPTMPO, b::UniformPTMPO)
    @assert a.s_dim == b.s_dim "system dimensions should be equal"

    if a.delta_t > b.delta_t
        a, b = b, a
    end

    if b.delta_t ≈ 2 * a.delta_t
        aq_half = a.q
    else
        @assert a.delta_t ≈ b.delta_t "time steps should be equal"
        if bond_dim(a) > bond_dim(b)
            a, b = b, a
        end
        aq_half = (a^0.5).q
    end

    @tensoropt q[-1, -2, -3, -4, -5, -6] := aq_half[-1, -3, 1, 2] * b.q[-2, 2, -5, 3] * aq_half[1, 3, -4, -6]
    q = reshape(q, bond_dim(a) * bond_dim(b), a.s_dim^2, bond_dim(a) * bond_dim(b), a.s_dim^2)
    @tensoropt v_l[n, m] := (a.v_l[:])[n] * (b.v_l[:])[m]
    @tensoropt v_r[n, m] := a.v_r[n] * b.v_r[m]
    return UniformPTMPO(a.s_dim, a.delta_t, q, v_r[:], transpose(v_l[:]))
end


"""
    combine_no_trotter(pt1::UniformPTMPO, pt2::UniformPTMPO)

same as `+` but formally avoids Trotter errors by re-exponentiation. Can be unstable because of the matrix logarithm.
"""
function combine_no_trotter(pt1::UniformPTMPO, pt2::UniformPTMPO)
    #TODO: test this function
    @assert pt1.delta_t == pt2.delta_t "Time steps are not equal."
    @assert pt1.s_dim == pt2.s_dim "system dimensions should be equal"

    g1 = reshape(kron(I(size(pt2.q, 2)), log(reshape(pt1.q, size(pt1.q, 1) * size(pt1.q, 2), :))), size(pt1.q, 1) * size(pt2.q, 2) * size(pt1.q, 2), :)
    g2 = reshape(permutedims(reshape(kron(I(size(pt1.q, 2)), log(reshape(pt2.q, size(pt2.q, 1) * size(pt2.q, 2), :))), size(pt2.q, 1), size(pt2.q, 2), size(pt1.q, 2), size(pt2.q, 1), size(pt2.q, 2), size(pt1.q, 2)), [1, 3, 2, 4, 6, 5]), size(pt1.q, 1) * size(pt2.q, 2) * size(pt1.q, 2), :)
    return UniformPTMPO(pt1.s_dim, pt1.delta_t, reshape(exp(reshape(g1 + g2, size(pt1.q, 1) * size(pt2.q, 2) * size(pt1.q, 2), :)), size(pt1.q, 1), size(pt2.q, 2) * size(pt1.q, 2), size(pt1.q, 1), size(pt2.q, 2) * size(pt1.q, 2)), kron(pt2.v_r, pt1.v_r), transpose(kron(pt2.v_l, pt1.v_l)[:]))
end


"""
    ^(pt::UniformPTMPO, y::Real)

Take the power of a uniform process tensor to change the time step.
"""
function ^(pt::UniformPTMPO, y::Real)
    q = reshape(pt.q, size(pt.q, 1) * size(pt.q, 2), :)
    return UniformPTMPO(pt.s_dim, pt.delta_t * y, reshape(q^y, size(pt.q)), pt.v_r, pt.v_l)
end


"""
    *(pt::UniformPTMPO, Φ::AbstractMatrix{<:Number})

Combine a process tensor with a channel Φ on the right.
"""
function *(pt::UniformPTMPO, Φ::AbstractMatrix{<:Number})
    @assert size(Φ) == (pt.s_dim^2, pt.s_dim^2) "Channel shape inconsistent."
    q = reshape(reshape(pt.q, :, size(pt.q, 4)) * Φ, size(pt.q))
    return UniformPTMPO(pt.s_dim, pt.delta_t, q, pt.v_r, pt.v_l)
end


"""
    *(Φ::AbstractMatrix{<:Number}, pt::UniformPTMPO)

Combine a process tensor with a channel Φ on the left.
"""
function *(Φ::AbstractMatrix{<:Number}, pt::UniformPTMPO)
    @assert size(Φ) == (pt.s_dim^2, pt.s_dim^2) "Channel shape inconsistent."
    q = permutedims(reshape(Φ * reshape(permutedims(pt.q, [2, 1, 3, 4]), size(pt.q, 2), :), size(pt.q, 2), size(pt.q, 1), size(pt.q, 3), size(pt.q, 4)), [2, 1, 3, 4])
    return UniformPTMPO(pt.s_dim, pt.delta_t, q, pt.v_r, pt.v_l)
end


"""
    UniformPTMPO_from_channel(delta_t::Real, Φ::AbstractArray{<:Number,4}, ρ_0::AbstractMatrix{<:Number})

Create a uniform PT-MPO from a channel `Φ`. System dimension is inferred from the channel and the bath initial state `ρ_0`.
"""
function UniformPTMPO_from_channel(delta_t::Real, Φ::AbstractMatrix{<:Number}, ρ_0::AbstractMatrix{<:Number})
    e_dim = size(ρ_0, 1)
    s_dim = trunc(Int, trunc(Int, sqrt(size(Φ, 1))) / e_dim)
    @assert size(Φ, 1) == (s_dim * e_dim)^2 "Channel shape inconsistent."
    q = reshape(permutedims(reshape(Φ, e_dim, s_dim, e_dim, s_dim, e_dim, s_dim, e_dim, s_dim), [1, 3, 2, 4, 5, 7, 6, 8]), e_dim^2, s_dim^2, e_dim^2, s_dim^2)
    v_r = ρ_0[:]
    v_l = transpose(I(e_dim)[:]) # trace
    return UniformPTMPO(s_dim, delta_t, q, v_r, v_l)
end


"""
    UniformPTMPO_from_hamiltonian(delta_t::Real, H::AbstractMatrix{<:Number}, ρ_0::AbstractMatrix{<:Number})

Create a uniform PT-MPO from a system-bath Hamiltonian `H`. System dimension is inferred from the Hamiltonian and the bath initial state `ρ_0`.
"""
function UniformPTMPO_from_hamiltonian(delta_t::Real, H::AbstractMatrix{<:Number}, ρ_0::AbstractMatrix{<:Number})
    Φ = exp(im * kron(transpose(H), Matrix(I, size(H))) * delta_t - im * kron(Matrix(I, size(H)), H) * delta_t)
    return UniformPTMPO_from_channel(delta_t, Φ, ρ_0)
end


"""
    UniformPTMPO_from_GKSL(delta_t::Real, H::AbstractMatrix{<:Number}, ρ_0::AbstractMatrix{<:Number}, L_ops)

Create a uniform PT-MPO from a GKLS equation with Hamiltonian `H` and jump operators `L_ops`. System dimension is inferred from the Hamiltonian and the bath initial state `ρ_0`.
"""
function UniformPTMPO_from_GKSL(delta_t::Real, H::AbstractMatrix{<:Number}, L_ops, ρ_0::AbstractMatrix{<:Number})
    GKSL = im * kron(transpose(H), Matrix(I, size(H))) - im * kron(Matrix(I, size(H)), H)
    Id = Matrix(I, size(H))
    for l in L_ops
        @assert size(l) == size(H) "Jump operator has invalid shape."
        GKSL += kron(transpose(l'), l) - 0.5 * kron(transpose(l' * l), Id) - 0.5 * kron(Id, l' * l)
    end
    return UniformPTMPO_from_channel(delta_t, exp(GKSL * delta_t), ρ_0)
end


"""
    prepare_initial_state(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number})

check initial state and prepare bath+system product state in case `ρ_0` is a system density matrix.
"""
function prepare_initial_state(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number})
    (size(ρ_0) == (pt.s_dim, pt.s_dim)) && (ρ_0 = pt.v_r * transpose(ρ_0[:]))
    @assert size(ρ_0) == (bond_dim(pt), pt.s_dim^2) "Initial state has invalid shape."
    return ρ_0
end


"""
    local_channel(t_span::Tuple{<:Number,<:Number}, h_s::Function)

Compute the local channel generated by a time-dependent Hamiltonian.
"""
function local_channel(t_span::Tuple{<:Number,<:Number}, h_s::Function)
    return local_channel(t_span, h_s, local_channel_default_backend[])
end
function local_channel(t_span::Tuple{<:Number,<:Number}, h_s::Function, ::LocalChannelBasic)
    delta_t = t_span[2] - t_span[1]
    h_s_ = h_s((t_span[1] + t_span[2]) / 2)
    return kron(transpose(exp(im * h_s_ * delta_t)), (exp(-im * h_s_ * delta_t)))
end


"""
    evolve(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing, AbstractMatrix{<:Number}, Function}=nothing, return_full::Bool=false)

Compute the time evolution for `n` time steps with initial state `ρ_0` (density matrix). `h_s` can be an additional systm Hamiltonian (may also be a time-dependent Hamiltonian).

The function simply returns a vector of density matrices at each time step.

# Example
```
ρ_t = evolve(pt, n, ρ_0; h_s=h_s)
t_eval = collect(0:n) * pt.delta_t
plot(t_eval, real.([tr(O * ρ_t[i]) for i in eachindex(ρ_t)])) # plot the dynamics of ⟨O⟩ = tr(O*rho)
```
"""
function evolve(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing, return_full::Bool=false)
    if typeof(h_s) <: Function
        return evolve(pt, ρ_0, n, h_s, return_full=return_full)
    end
    state = prepare_initial_state(pt, ρ_0)
    q = reshape(include_system_hamiltonian(pt, h_s).q, size(pt.q, 1) * size(pt.q, 2), size(pt.q, 1) * size(pt.q, 2))

    ρ_t = Vector{Matrix{eltype(pt)}}(undef, n + 1)
    ρ_t[1] = reshape(pt.v_l * state, pt.s_dim, pt.s_dim)

    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        state = reshape(q * state[:], size(state))
        ρ_t[i+1] = reshape(pt.v_l * state, pt.s_dim, pt.s_dim)
    end

    return_full && (return state)
    return ρ_t
end
function evolve(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::Int, h_s::Function; return_full::Bool=false)
    @assert typeof(h_s(0)) <: AbstractMatrix{<:Number} "Function h_s must return a matrix."
    @assert size(h_s(0)) == (pt.s_dim, pt.s_dim) "Hamiltonian has invalid shape."
    state = prepare_initial_state(pt, ρ_0)

    ρ_t = Vector{Matrix{eltype(pt)}}(undef, n + 1)
    ρ_t[1] = reshape(pt.v_l * state, pt.s_dim, pt.s_dim)
    q = reshape(pt.q, size(pt.q, 1) * size(pt.q, 2), size(pt.q, 1) * size(pt.q, 2))

    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        u1 = local_channel((pt.delta_t * (i - 1), pt.delta_t * (i - 0.5)), h_s)
        u2 = local_channel((pt.delta_t * (i - 0.5), pt.delta_t * (i)), h_s)
        state = state * transpose(u1)
        state = reshape(q * state[:], size(state))
        state = state * transpose(u2)
        ρ_t[i+1] = reshape(pt.v_l * state, pt.s_dim, pt.s_dim)
    end

    return_full && (return state)
    return ρ_t
end


"""
    pt_to_choi(pt::AbstractArray{<:Number})

Convert a process tensor (from the `process_tensor` function) to a Choi state. Returns the choi state in the form 

```math
ρ_{choi} = Σ_{i,i'} Y[i_0,i_1,...,i_N;i_0',i_1',...,i_N'] |i_0,i_1,...,i_N⟩ ⟨i_0',i_1',...,i_N'|
```

where `N` in the number of iterventions. `i_0` corresponds to th^e input state (if applicable), see also the docstring for `process_tensor`. For details on the Choi mapping see [Pollock et.al., Phys. Rev. A 97, 012127 (2018)].
"""
function pt_to_choi(pt::AbstractArray{<:Number})
    d = trunc(Int, sqrt(size(pt, 1)))
    n = length(size(pt))
    pt = reshape(pt, fill(d, 2 * n)...)

    if iseven(n)
        n_int = trunc(Int, (n - 2) / 2)
        perm = vcat([1], [[3 + (k - 1) * 4, 3 + (k - 1) * 4 + 2] for k in 1:n_int]..., [2 * n - 1, 2], [[3 + (k - 1) * 4 + 1, 3 + (k - 1) * 4 + 3] for k in 1:n_int]..., [2 * n])
        return reshape(permutedims(pt, perm) / d^(trunc(Int, n / 2)), d, fill(d^2, n_int)..., d, d, fill(d^2, n_int)..., d)
    else
        n_int = trunc(Int, (n - 1) / 2)
        perm = vcat([[1 + (k - 1) * 4, 1 + (k - 1) * 4 + 2] for k in 1:n_int]..., [2 * n - 1], [[1 + (k - 1) * 4 + 1, 1 + (k - 1) * 4 + 3] for k in 1:n_int]..., [2 * n])
        return reshape(permutedims(pt, perm) / d^(trunc(Int, n / 2)), fill(d^2, n_int)..., d, fill(d^2, n_int)..., d)
    end

end


"""
    channel_to_choi(x::AbstractArray{<:Number,4})

Convert a channel `x` in generic basis to a choi state.
"""
function channel_to_choi(x::AbstractMatrix{<:Number})
    d = trunc(Int, sqrt(size(x, 1)))
    @assert size(x) == (d^2, d^2) "Channel has invalid shape."
    return reshape(permutedims(reshape(x, d, d, d, d), [1, 3, 2, 4]), d^2, d^2) / d
end


"""
    channel(pt::UniformPTMPO, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)

compute dynamical map for `n` time-steps in generic basis (linear map on density matrix). `h_s` is an optional additional system Hamiltonian.
The map can be applied as follows

# Example
```
ρ_t = evolve(pt, ρ_0, n)
Φ_t = channel(pt, n)
ρ̃_t = [reshape(Φ_t[i] * ρ_0[:], size(ρ_0)) for i in eachindex(Φ_t)]
ρ_t ≈ ρ̃_t # true
```
"""
function channel(pt::UniformPTMPO, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)
    if typeof(h_s) <: Function
        return channel(pt, n, h_s)
    end

    q = reshape(include_system_hamiltonian(pt, h_s).q, size(pt.q, 1) * size(pt.q, 2), size(pt.q, 1) * size(pt.q, 2))

    Φ_t = Vector{Matrix{eltype(pt)}}(undef, n + 1)
    #zeros(eltype(pt), n + 1, pt.s_dim^2, pt.s_dim^2)
    Φ_t[1] = Matrix(I, pt.s_dim^2, pt.s_dim^2)
    @tensor state[-1, -2, -3] := pt.v_r[-1] * Matrix(I, pt.s_dim^2, pt.s_dim^2)[-2, -3]
    state = reshape(state, :, pt.s_dim^2)

    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        state = q * state
        Φ_t[i+1] = reshape(pt.v_l * reshape(state, :, pt.s_dim^4), pt.s_dim^2, pt.s_dim^2)
    end
    return Φ_t
end
function channel(pt::UniformPTMPO, n::Int, h_s::Function)
    @assert size(h_s(0)) == (pt.s_dim, pt.s_dim) "Hamiltonian has invalid shape."

    Φ_t = Vector{Matrix{eltype(pt)}}(undef, n + 1)
    Φ_t[1] = Matrix(I, pt.s_dim^2, pt.s_dim^2)

    @tensor state[-1, -2, -3] := pt.v_r[-1] * Matrix(I, pt.s_dim^2, pt.s_dim^2)[-2, -3]
    state = reshape(state, :, pt.s_dim^4)
    q = reshape(pt.q, size(pt.q, 1) * size(pt.q, 2), size(pt.q, 1) * size(pt.q, 2))
    @showprogress showspeed = true desc = "performing time evolution..." for i in 1:n
        u1 = local_channel((pt.delta_t * (i - 1), pt.delta_t * (i - 0.5)), h_s)
        u2 = local_channel((pt.delta_t * (i - 0.5), pt.delta_t * (i)), h_s)
        state = state * transpose(kron(I(pt.s_dim^2), u1))
        state = reshape(q * reshape(state, :, pt.s_dim^2), size(state))
        state = state * transpose(kron(I(pt.s_dim^2), u2))
        Φ_t[i+1] = reshape(pt.v_l * state, pt.s_dim^2, pt.s_dim^2)
    end
    return Φ_t
end


"""
    choi_channel(pt::UniformPTMPO, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)

compute the dynamical map for given hamiltonian `h_s` for `n` time-steps in choi representation.
"""
function choi_channel(pt::UniformPTMPO, n::Int; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)
    Φ_t = channel(pt, n; h_s=h_s)
    return [channel_to_choi(Φ) for Φ in Φ_t]
end


"""
    steadystate(pt::UniformPTMPO; h_s::AbstractMatrix{<:Number}=zeros(pt.s_dim, pt.s_dim), return_full::Bool=false, ED::Bool=false)

Compute the steady state of a process tensor `pt`. An additional system Hamiltonian `h_s` can be specified directly for convenience.

Returns the system steady state if `return_full=false` and the full leading eigenvector of q if `return_full=true`. Set `ED=true` to use exact diagonalization instead of an iterative eigensolver.
"""
function steadystate(pt::UniformPTMPO; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing, return_full::Bool=false, ED::Bool=false)
    if ED
        _, R = spectrum(pt; h_s=h_s)
        v = R[:, 1]
    else
        q = include_system_hamiltonian(pt, h_s).q
        _, vecs, _ = eigsolve(reshape(q, size(q, 1) * size(q, 2), size(q, 1) * size(q, 2)), 1, :LR; maxiter=1000)
        v = vecs[1]
    end
    ρ_ss = reshape(pt.v_l * reshape(v, :, pt.s_dim^2), pt.s_dim, pt.s_dim)

    return_full ? (return reshape(v, :, pt.s_dim^2) / tr(ρ_ss)) : (return ρ_ss / tr(ρ_ss))
end


"""
    expect(o::AbstractArray{<:Number,2}, ρ_t::AbstractArray{<:Number,3})
    expect(o::AbstractArray{<:Number,2}, ρ::AbstractArray{<:Number,2})

convenience function to compute expectation values.
"""
function expect(o::AbstractArray{<:Number,2}, ρ_t::AbstractVector)
    return [expect(o, ρ_t[i]) for i in eachindex(ρ_t)]
end
function expect(o::AbstractArray{<:Number,2}, ρ::AbstractArray{<:Number,2})
    return tr(ρ * o)
end


"""
    two_point_correlations(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n_1::Int, n_2::Int, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}; h_s::Union{Nothing,AbstractMatrix{<:Number},Function} = nothing)


Calculate the two-time correlation function ``⟨o₂(t+s)o₁(s)⟩`` for operators `o_1`, `o_2`. `h_s` is the system Hamiltonian, `n_1` is the number of initial time steps where `s = pt.delta_t * n_1` and `n_2` the number of evolution time steps `t = pt.delta_t * n_2`. Returns the correlation function for all `n <= n_2`. By default, the operators are left acting (Keldysh "+" contour). For right acting operators directly provide the superoperators.


    two_point_correlations(pt::UniformPTMPO, n::Int, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}; h_s::Union{Nothing,AbstractMatrix{<:Number},Function} = nothing)

Calculate the stationary two-time correlation function ``⟨o₂(t)o₁(0)⟩`` for operators `o_1`, `o_2`. `h_s` is the system Hamiltonian and `n` is the number of time steps (as in `evolve`). 

"""
function two_point_correlations(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n_1::Int, n_2::Int, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)
    if typeof(h_s) <: Function
        return two_point_correlations(pt, ρ_0, n_1, n_2, o_1, o_2, h_s)
    end
    (size(o_1) == (pt.s_dim, pt.s_dim)) && (o_1 = kron(I(pt.s_dim), o_1)) # use left-acting if not superoperator
    (size(o_2) == (pt.s_dim, pt.s_dim)) && (o_2 = kron(I(pt.s_dim), o_2))
    @assert size(o_1) == (pt.s_dim^2, pt.s_dim^2) "Operator 1 has incompatible shape."
    @assert size(o_2) == (pt.s_dim^2, pt.s_dim^2) "Operator 2 has incompatible shape."

    q = reshape(include_system_hamiltonian(pt, h_s).q, size(pt.q, 1) * size(pt.q, 2), size(pt.q, 1) * size(pt.q, 2))

    # apply second operator and trace 
    trace_tensor = transpose(o_2) * (Matrix(I, pt.s_dim, pt.s_dim)[:])

    # correlations depending on time s (n_2)
    cf = zeros(eltype(q), n_2 + 1)

    # evolve to n_1 and apply first operator
    state = reshape(q^n_1 * prepare_initial_state(pt, ρ_0)[:], bond_dim(pt), pt.s_dim^2) * transpose(o_1)

    #zeroth step
    cf[1] = pt.v_l * state * trace_tensor

    #evolution up to n_2
    @showprogress showspeed = true desc = "calculating correlation function..." for i in 1:n_2
        # evolve for one step
        state = reshape(q * (state[:]), size(state))
        cf[i+1] = pt.v_l * state * trace_tensor
    end

    return cf
end
function two_point_correlations(pt::UniformPTMPO, n::Int, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    x0 = steadystate(pt; h_s=h_s, return_full=true)
    return two_point_correlations(pt, x0, 0, n, o_1, o_2; h_s)
end
function two_point_correlations(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n_1::Int, n_2::Int, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, h_s::Function)
    (size(o_1) == (pt.s_dim, pt.s_dim)) && (o_1 = kron(I(pt.s_dim), o_1))
    (size(o_2) == (pt.s_dim, pt.s_dim)) && (o_2 = kron(I(pt.s_dim), o_2))
    @assert size(o_1) == (pt.s_dim^2, pt.s_dim^2) "Operator 1 has incompatible shape."
    @assert size(o_2) == (pt.s_dim^2, pt.s_dim^2) "Operator 2 has incompatible shape."

    # apply second operator and trace 
    trace_tensor = transpose(o_2) * (Matrix(I, pt.s_dim, pt.s_dim)[:])

    # correlations
    cf = zeros(eltype(pt), n_2 + 1)

    # get initial state and apply first operator
    state = prepare_initial_state(pt, ρ_0)

    q = reshape(pt.q, size(pt.q, 1) * size(pt.q, 2), size(pt.q, 1) * size(pt.q, 2))

    for i in 1:n_1
        u1 = local_channel((pt.delta_t * (i - 1), pt.delta_t * (i - 0.5)), h_s)
        u2 = local_channel((pt.delta_t * (i - 0.5), pt.delta_t * (i)), h_s)
        state = state * transpose(u1)
        state = reshape(q * state[:], size(state))
        state = state * transpose(u2)
    end
    state = state * transpose(o_1)

    #zeroth step
    cf[1] = pt.v_l * state * trace_tensor

    #evolution
    @showprogress showspeed = true desc = "calculating correlation function..." for i in (n_1+1):(n_1+n_2)
        # evolve for one step
        u1 = local_channel((pt.delta_t * (i - 1), pt.delta_t * (i - 0.5)), h_s)
        u2 = local_channel((pt.delta_t * (i - 0.5), pt.delta_t * (i)), h_s)
        state = state * transpose(u1)
        state = reshape(q * state[:], size(state))
        state = state * transpose(u2)
        cf[1+i-n_1] = pt.v_l * state * trace_tensor
    end

    return cf
end


"""
    process_tensor(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::AbstractVector{<:Int}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)

compute the process tensor in canonical (Liouville-space) basis for interventions at time steps `n`. Returns the process tensor in the form 

```math
T[(μ^1_I,μ^1_O),(μ^2_I,μ^2_O),...,μ^N_I]
```

where `N=length(n)` and ``μ^i_{I/0}=1...d^2``. The trivial output index for the last time step is omitted. If no initial state is provided, the output index for the initial state is included as a first index. To convert a process tensor of this form to choi representation, use `pt_tensor_to_choi`. See [Pollock et.al., Phys. Rev. A 97, 012127 (2018)] for details.
"""
function process_tensor(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, n::AbstractVector{<:Int}; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)
    @assert all(diff(n) .> 0) "Intervention time steps n must be ordered."
    state = prepare_initial_state(pt, ρ_0)
    if typeof(h_s) <: Function
        throw("Time-dependent Hamiltonians are not yet implemented for the process_tensor function.")
    end
    q = reshape(include_system_hamiltonian(pt, h_s).q, size(pt.q, 1) * size(pt.q, 2), :)
    pt_ = reshape(q^n[1] * state[:], size(state))

    for i in 2:lastindex(n)
        pt_ = reshape(pt_, :, size(pt.q, 1), pt.s_dim^2)
        @tensoropt pt_new[a, b, c, x, d] := reshape(q^(n[i] - n[i-1]), size(pt.q, 1), pt.s_dim^2, size(pt.q, 1), pt.s_dim^2)[x, d, y, c] * pt_[a, y, b]
        pt_ = pt_new
    end
    pt_ = reshape(pt_, :, size(pt.q, 1), pt.s_dim^2)
    @tensoropt pt_new[a, b] := pt_[a, x, b] * (pt.v_l[:])[x]
    return reshape(pt_new, Tuple([pt.s_dim^2 for i in 1:(2*length(n)-1)]))
end
function process_tensor(pt::UniformPTMPO, n::AbstractVector{<:Int}; h_s::Union{Nothing,AbstractMatrix{<:Number},Function}=nothing)
    @assert all(diff(n) .> 0) "Intervention time steps n must be ordered."
    if typeof(h_s) <: Function
        throw("Time-dependent Hamiltonians are not yet implemented for the process_tensor function.")
    end

    q = reshape(include_system_hamiltonian(pt, h_s).q, size(pt.q, 1) * size(pt.q, 2), :)
    @tensoropt pt_[a, x, b] := pt.v_r[y] * reshape(q^(n[1]), size(pt.q, 1), pt.s_dim^2, size(pt.q, 1), pt.s_dim^2)[x, b, y, a]

    for i in 2:lastindex(n)
        pt_ = reshape(pt_, :, size(pt.q, 1), pt.s_dim^2)
        @tensoropt pt_new[a, b, c, x, d] := reshape(q^(n[i] - n[i-1]), size(pt.q, 1), pt.s_dim^2, size(pt.q, 1), pt.s_dim^2)[x, d, y, c] * pt_[a, y, b]
        pt_ = pt_new
    end
    pt_ = reshape(pt_, :, size(pt.q, 1), pt.s_dim^2)
    @tensoropt pt_new[a, b] := pt_[a, x, b] * (pt.v_l[:])[x]
    return reshape(pt_new, Tuple([pt.s_dim^2 for i in 1:(2*length(n))]))
end


"""
    spectrum(pt::UniformPTMPO; h_s::AbstractMatrix{<:Number}=zeros(pt.s_dim, pt.s_dim); enforce_physical=false)

Compute the spectrum of the process tensor transfer matrix for Hamiltonian `h_s` using full diagonalization. 

When setting `enforce_physical=true` the dominant decay rate is truncated to zero explicitly.

Returns the spectrum `γ` and the basis change matrix `R` such that `q(t)=R*exp.(diagm(γ)*t)*inv(R)`, where `q(t)` is the propagator for time `t`.
"""
function spectrum(pt::UniformPTMPO; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing, enforce_physical=false)
    q = include_system_hamiltonian(pt, h_s).q

    λ, R = eigen(reshape(q, size(q, 1) * size(q, 2), size(q, 1) * size(q, 2)))

    γ = log.(λ) / pt.delta_t

    # sort eigenvalues
    order = sortperm(abs.(real.(γ)))

    γ = γ[order]
    if enforce_physical
        γ[1] = 0
        γ[real.(γ).>0] .= im * imag(γ[real.(γ).>0])
    end

    R = R[:, order]

    return γ, R
end


"""
    two_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)


Directly compute the half-sided fourier transform of the stationary two point function ``c(t) = ⟨o_2(t)o_1(0)⟩`` evalutated at frequencies ω. 

    two_point_correlations_fourier(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)

If no initial state is passed then this computes the stationary correlation function.

This function accepts both superoperators and normal operators. Normal operators are always interpreted as left-acting. For right-acting operators, provide the corresponding superoperator `o_right = kron(transpose(o), I(size(o, 1)))`.
"""
function two_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    (size(o_1) == (pt.s_dim, pt.s_dim)) && (o_1 = kron(I(pt.s_dim), o_1))
    (size(o_2) == (pt.s_dim, pt.s_dim)) && (o_2 = kron(I(pt.s_dim), o_2))

    @assert size(o_1) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_1` has invalid shape."
    @assert size(o_2) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_2` has invalid shape."
    ss = prepare_initial_state(pt, ρ_0)

    γ, r = spectrum(pt; h_s=h_s, enforce_physical=false)
    rinv = reshape(inv(r), size(r, 1), size(pt.q, 1), pt.s_dim^2)
    r = reshape(r, size(pt.q, 1), pt.s_dim^2, size(r, 1))

    c1 = zeros(eltype(pt), size(ω, 1))

    # pre-compute boundaries
    s̃ = (ss*transpose(o_1))[:]
    õ_2 = (I(pt.s_dim)[:]'*o_2)[:]

    for k in eachindex(γ)
        _c1 = transpose((rinv[k, :, :])[:]) * s̃
        _c2 = pt.v_l * (r[:, :, k]) * õ_2
        c1 += _c1 * _c2 * (@. (-1) / (im * ω + γ[k]))
    end

    return c1
end
function two_point_correlations_fourier(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    two_point_correlations_fourier(pt, steadystate(pt; h_s=h_s, return_full=true), o_1, o_2, ω; h_s=h_s)
end


"""
    two_point_correlations_fourier_schur(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}} = nothing)


Same as `two_point_correlations_fourier` but using Schur decomposition instead of diagonalization. More costly but guaranteed stability. Requires initial state as argument. 
"""
function two_point_correlations_fourier_schur(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    (size(o_1) == (pt.s_dim, pt.s_dim)) && (o_1 = kron(I(pt.s_dim), o_1))
    (size(o_2) == (pt.s_dim, pt.s_dim)) && (o_2 = kron(I(pt.s_dim), o_2))

    @assert size(o_1) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_1` has invalid shape."
    @assert size(o_2) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_2` has invalid shape."
    ss = prepare_initial_state(pt, ρ_0)

    pt = include_system_hamiltonian(pt, h_s)

    T, Q = schur(log(reshape(pt.q, size(pt.q, 1) * size(pt.q, 2), :)) / pt.delta_t) # schur decomposition of generator

    c1 = zeros(eltype(pt), size(ω, 1))
    # pre-compute boundaries
    sd = (Q'*((ss*transpose(o_1))[:]))[:]
    ad = (transpose((transpose(pt.v_l)*(I(pt.s_dim)[:]'*o_2))[:])*Q)[:]

    for i in eachindex(ω)
        c1[i] = transpose(ad) * (UpperTriangular(-T + im * ω[i] * I) \ sd)
    end

    return c1
end


"""
    susceptibility(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)


Directly compute the fourier transform of the stationary susceptibility ``χ(t-s) = iΘ(t-s) ⟨[o_2(t),o_1(s)]⟩`` evalutated at frequencies `ω`.
"""
function susceptibility(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, ω::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    o_2_super = kron(I(pt.s_dim), o_2)
    o_1_super = kron(I(pt.s_dim), o_1) - kron(transpose(o_1), I(pt.s_dim))
    c1 = two_point_correlations_fourier(pt, o_1_super, o_2_super, ω; h_s=h_s)
    return im * (c1)
end


"""
    three_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, ω_23::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)


Directly compute the half-sided fourier transform of the three point function `R(t, s) = ⟨o_3(t)o_2(s)o_1(0)⟩` evalutated at frequencies `ω_12` (time `s`) and `ω_23` (time `t`). 


    three_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, ω_23::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)

Stationary version that uses the steadystate as initial state.
"""
function three_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, ω_23::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    return four_point_correlations_fourier(pt, ρ_0, o_1, o_2, Matrix(I, size(o_2)), o_3, ω_12, 0, ω_23; h_s=h_s)

    # this older implementation is slow, thats why we just call the four point function with zero waiting time.

    @assert size(o_1) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_1` has invalid shape."
    @assert size(o_2) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_2` has invalid shape."
    @assert size(o_3) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_3` has invalid shape."

    γ, r = spectrum(pt; h_s=h_s, enforce_physical=false)
    rinv = reshape(inv(r), size(r, 1), size(pt.q, 1), pt.s_dim^2)
    r = reshape(r, size(pt.q, 1), pt.s_dim^2, size(r, 1))
    ss = prepare_initial_state(pt, ρ_0)

    c1 = zeros(eltype(pt), size(ω_12, 1), size(ω_23, 1))

    # pre-compute boundaries and coefficients
    s̃ = (ss*transpose(o_1))[:]
    õ_3 = (I(pt.s_dim)[:]'*o_3)[:]
    _c1k = zeros(eltype(pt), size(r, 3))
    _c3k = zeros(eltype(pt), size(r, 3))

    for k in eachindex(γ)
        _c1k[k] = transpose((rinv[k, :, :])[:]) * s̃
        _c3k[k] = pt.v_l * (r[:, :, k]) * õ_3
    end

    @showprogress desc = "computing the response function..." for (k1, k2) in Base.Iterators.product(eachindex(γ), eachindex(γ))
        #_c1 = transpose((rinv[k1, :, :])[:]) * s̃
        _c2 = transpose((rinv[k2, :, :]*o_2)[:]) * (r[:, :, k1])[:]
        #_c3 = pt.v_l * (r[:, :, k2]) * ã
        c1 += _c1k[k1] * _c2 * _c3k[k2] * (@. (-1) / (im * ω_23 + γ[k2])) * transpose(@. (-1) / (im * ω_12 + γ[k1]))
    end

    return c1
end
function three_point_correlations_fourier(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, ω_23::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    return three_point_correlations_fourier(pt, steadystate(pt; h_s=h_s, return_full=true), o_1, o_2, o_3, ω_12, ω_23; h_s=h_s)
end


"""
    four_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, o_4::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, n_23::Int, ω_34::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)


Directly compute the half-sided fourier transform of the four point correlation function R(t, τ) = ⟨o_4(t)o_3(τ+T)o_2(τ)o_1(0)⟩ evalutated at frequencies ω_12 (τ) and ω_34 (t), with waiting time T (n_23 time-steps). 


    four_point_correlations_fourier(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, o_4::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, n_23::Int, ω_34::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)


Stationary version that uses the steadystate as initial state.
"""
function four_point_correlations_fourier(pt::UniformPTMPO, ρ_0::AbstractMatrix{<:Number}, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, o_4::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, n_23::Int, ω_34::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)

    @assert size(o_1) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_1` has invalid shape."
    @assert size(o_2) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_2` has invalid shape."
    @assert size(o_3) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_3` has invalid shape."
    @assert size(o_4) == (pt.s_dim^2, pt.s_dim^2) "Operator `o_4` has invalid shape."
    @assert size(h_s) == (pt.s_dim, pt.s_dim) "Hamiltonian has invalid shape."
    @assert size(ω_12) == size(ω_34) "Frequency arrays `ω_12` and `ω_34` have different lenghts. Current implementation requires the same number of sampling points."

    γ, r = spectrum(pt; h_s=h_s, enforce_physical=false)

    # initial and boundary states
    ss = prepare_initial_state(pt, ρ_0)
    s̃ = (ss*transpose(o_1))[:]
    õ_4 = (I(pt.s_dim)[:]'*o_4)[:]

    # waiting-time propagator via eigendecomposition
    qn = r * Diagonal(exp.(γ .* pt.delta_t) .^ n_23) / r

    # reshape eigenvectors for contractions
    rinv = reshape(inv(r), size(r, 1), size(pt.q, 1), pt.s_dim^2)
    r = reshape(r, size(pt.q, 1), pt.s_dim^2, size(r, 1))

    # transposed operators for the inner contractions
    oᵀ_3 = transpose(o_3)
    oᵀ_2 = transpose(o_2)

    # pre-compute coefficients 
    _c1k = zeros(eltype(pt), size(r, 3))
    _c3k = zeros(eltype(pt), size(r, 3))
    for k in eachindex(γ)
        _c1k[k] = transpose((rinv[k, :, :])[:]) * s̃
        _c3k[k] = pt.v_l * (r[:, :, k]) * õ_4
    end

    # output array 
    C = zeros(eltype(pt), size(ω_12, 1), size(ω_34, 1))

    # lorentzian peak contributions (fourier transform)
    D1 = @. (-1) / (im * ω_12' + γ)
    D2 = @. (-1) / (im * ω_34' + γ)

    # pre-allocate for efficient loop evaluation
    tmp_rc = Matrix{eltype(qn)}(undef, size(pt.q, 1), pt.s_dim^2)
    tmp_v1 = Vector{eltype(qn)}(undef, size(pt.q, 1) * pt.s_dim^2)
    tmp_v2 = similar(tmp_v1)
    tmp_m1 = similar(tmp_rc)
    tmp_m2 = similar(tmp_rc)
    tmp_v = similar(tmp_v1)


    @views @showprogress desc = "computing the response function..." for k1 in eachindex(γ)

        mul!(tmp_rc, r[:, :, k1], oᵀ_2)
        tmp_v1 .= vec(tmp_rc)
        mul!(tmp_v2, qn, tmp_v1)
        tmp_m1 .= reshape(tmp_v2, size(pt.q, 1), pt.s_dim^2)
        mul!(tmp_m2, tmp_m1, oᵀ_3)
        tmp_v .= vec(tmp_m2)

        for k2 in eachindex(γ)
            _c2 = transpose(vec(rinv[k2, :, :])) * tmp_v
            α = _c1k[k1] * _c2 * _c3k[k2]
            BLAS.geru!(α, D2[k2, :], D1[k1, :], C)
        end
    end

    return C
end
function four_point_correlations_fourier(pt::UniformPTMPO, o_1::AbstractMatrix{<:Number}, o_2::AbstractMatrix{<:Number}, o_3::AbstractMatrix{<:Number}, o_4::AbstractMatrix{<:Number}, ω_12::AbstractVector{<:Real}, n_23::Int, ω_34::AbstractVector{<:Real}; h_s::Union{Nothing,AbstractMatrix{<:Number}}=nothing)
    return four_point_correlations_fourier(pt, steadystate(pt; h_s=h_s, return_full=true), o_1, o_2, o_3, o_4, ω_12, n_23, ω_34; h_s=h_s)
end


"""
    floquet_process_tensor(pt::UniformPTMPO, h_s::Function, Δt::Real)

Construct a Floquet process tensor from a uniform PT-MPO and a periodic system Hamiltonian `h_s`. The period is `Δt`. 
"""
function floquet_process_tensor(pt::UniformPTMPO, h_s::Function, Δt::Real)
    @assert size(h_s(0.0)) == (pt.s_dim, pt.s_dim) "System Hamiltonian has inconsistent dimensions."
    m = round(Int, Δt / pt.delta_t) # find closest commensurate
    if !(Δt / (pt.delta_t * m) ≈ 1)
        pt = pt^(Δt / (pt.delta_t * m)) # change process tensor time step to be commensurate
    end
    qptf = I(size(pt.q, 1) * size(pt.q, 2))
    #TODO: add an option for specifing Floquet gauge
    for i in 1:m
        u1 = local_channel((pt.delta_t * (i - 1), pt.delta_t * (i - 0.5)), h_s)
        u2 = local_channel((pt.delta_t * (i - 0.5), pt.delta_t * (i)), h_s)
        qptf = reshape((u2 * pt * u1).q, size(qptf)) * qptf
    end
    return UniformPTMPO(pt.s_dim, Δt, reshape(qptf, size(pt.q)), pt.v_r, pt.v_l)
end


"""
    orthogonalize(pt::UniformPTMPO; pre_svd::Bool=true)

Bring a UniformPTMPO to orthogonal form. Returns orthogonal UniformPTMPO in truncatable form and singular values corresponding to the bond index.
"""
function orthogonalize(pt::UniformPTMPO; pre_svd::Bool=true)
    x = zeros(eltype(pt.q), size(pt.q, 1), size(pt.q, 2) * size(pt.q, 4) + 1, size(pt.q, 3))
    x[:, 1:end-1, :] = reshape(permutedims(pt.q, [1, 2, 4, 3]), size(pt.q, 1), size(pt.q, 2) * size(pt.q, 4), size(pt.q, 3))
    @tensoropt x0[-1, -2] := pt.v_r[-1] * pt.v_l[:][-2]
    x[:, end, :] = x0

    if pre_svd
        u, s_vals, v = svd(reshape(permutedims(x, [1, 3, 2]), size(x, 1) * size(x, 3), size(x, 2)))
        rank_new = new_rank(s_vals, 1e-14, :rel)
        if rank_new < length(s_vals)
            u = u[:, 1:rank_new] * diagm(s_vals[1:rank_new])
            v = v'[1:rank_new, :]
            x̃ = permutedims(reshape(u[:, 1:rank_new], size(x, 1), size(x, 3), rank_new), [1, 3, 2])
            x̃, _, _, c = mixed_canonical(x̃)
            @tensoropt x_new[n, a, m] := x̃[n, b, m] * v[b, a]
        else
            x_new, _, _, c = mixed_canonical(x)
        end
    else
        x_new, _, _, c = mixed_canonical(x)
    end

    v_r, v_l = fixed_points(x_new[:, end, :])
    s_vals = diag(c)
    return UniformPTMPO(pt.s_dim, pt.delta_t, permutedims(reshape(x_new[:, 1:end-1, :], size(pt.q, 1), size(pt.q, 2), size(pt.q, 4), size(pt.q, 3)), [1, 2, 4, 3]), v_r, transpose(v_l)), s_vals
end


"""
    new_rank(s::AbstractVector{<:Real}, tol::Real, truncation::Symbol; cap_rank::Int=100_000)

Find a new rank for truncating ordered singular values `s` at a tolerance `tol` with truncation scheme `truncation`. Available are relative `:rel` and absolute `:àbs`.
"""
function new_rank(s::AbstractVector{<:Real}, tol::Real, truncation::Symbol; cap_rank::Int=100_000)
    if truncation == :rel
        s_sum = cumsum(s) ./ sum(s)
        rank_tol = searchsortedfirst(s_sum, 1 - tol)
        return minimum([length(s), rank_tol, cap_rank])
    end

    if truncation == :abs
        s = s / s[1]
        rank_abs = searchsortedfirst(-s, -tol) - 1
        return minimum([length(s), rank_abs, cap_rank])
    end

    throw(UndefVarError(truncation))
end


"""
    truncate(pt::UniformPTMPO, tol::Real; pre_svd::Bool=true)

truncate the bond dimension of a UniformPTMPO based on a tolerance `tol`.
"""
function truncate(pt::UniformPTMPO, tol::Real; pre_svd::Bool=true)
    pt, s = orthogonalize(pt; pre_svd=pre_svd)
    rank_new = new_rank(s, tol, :abs)
    return UniformPTMPO(pt.s_dim, pt.delta_t, pt.q[1:rank_new, :, 1:rank_new, :], pt.v_r[1:rank_new], transpose(pt.v_l[:][1:rank_new]))
end


"""
    triangular_combine(pt1::UniformPTMPO, pt2::UniformPTMPO, tol)

Add two environments while keeping only dominant singular values via preselection, as in [Cygorek et.al., Phys. Rev. Research 6, 043203 (2024)]. Requires doing an orthogonalization first.
"""
function triangular_combine(pt1::UniformPTMPO, pt2::UniformPTMPO, tol)
    @assert pt1.delta_t == pt2.delta_t "time steps should be equal"
    @assert pt1.s_dim == pt2.s_dim "system dimensions should be equal"
    pt1, s1 = orthogonalize(pt1)
    pt2, s2 = orthogonalize(pt2)
    tol = tol * s1[1] * s2[1]
    dim_k = sum([sv1 * sv2 > tol for sv1 in s1, sv2 in s2])
    combiner = zeros(Bool, dim_k, length(s1), length(s2))
    k = 1
    for i in eachindex(s1), j in eachindex(s2)
        if s1[i] * s2[j] > tol
            combiner[k, i, j] = 1
            k += 1
        end
    end
    pt2_half = pt2^0.5
    @tensor backend = TensorOperations.StridedBLAS() q[-2, -1, -4, -3] := (combiner[-2, 1, 2] * pt2_half.q[2, -1, 4, 3]) * pt1.q[1, 3, 6, 5] * (pt2_half.q[4, 5, 7, -3] * combiner[-4, 6, 7])
    @tensoropt vl[-1] := combiner[-1, 1, 2] * pt1.v_l[:][1] * pt2.v_l[:][2]
    @tensoropt vr[-1] := combiner[-1, 1, 2] * pt1.v_r[1] * pt2.v_r[2]

    return UniformPTMPO(pt1.s_dim, pt1.delta_t, q, vr, transpose(vl))
end