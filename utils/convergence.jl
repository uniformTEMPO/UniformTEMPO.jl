using UniformTEMPO
using LinearAlgebra
using JLD2

function convergence(   value_func::Function, S::AbstractMatrix{<:Number}, 
                        trotter::AbstractArray{<:Number}, bcf::Function, accuracy::AbstractArray{<:Number}; max_rank::Int=100_000,
                        path::String = pwd(),
                        checkpoint_file::String="checkpoint.jld2",
                        output_file::String="convergence_data.jld2",
                    )


    # validate user path
    @assert isdir(abspath(expanduser(path))) "Path does not exist: '$path'"

    # build file paths 
    endswith(output_file, ".jld2") || (output_file *= ".jld2")
    endswith(checkpoint_file, ".jld2") || (checkpoint_file *= ".jld2")
    checkpoint_path = joinpath(path, checkpoint_file)
    output_path = joinpath(path, output_file)


    # extract return type of value_func
    # benefit: passes all possible warning/errors to the base uniTEMPO function
    pt = uniTEMPO(S, trotter[1], bcf, accuracy[1])
    T = typeof(value_func(pt))

    # allocate bond_dimension, values, and last index arrays
    bond_dimensions = Array{Union{Int64, Nothing}}(nothing, length(trotter), length(accuracy));
    values = Array{Union{T, Nothing}}(nothing, length(trotter), length(accuracy));
    indices = Array{Int}(undef, length(trotter))


    # checkpoint closure 
    checkpoint(j, k) = (tmpfile = checkpoint_path * ".tmp";
    jldsave(tmpfile; bond_dimensions, values, trotter, accuracy, trotter_index=j, accuracy_index=k, indices);
    mv(tmpfile, checkpoint_path; force=true))

    # main convergence calculation
    for j in eachindex(trotter)
        for k in eachindex(accuracy)
            try
                MyPT = uniTEMPO(S, trotter[j], bcf, accuracy[k]; max_rank = max_rank)
                bond_dimensions[j, k]= bond_dim(MyPT)
                values[j, k] = value_func(MyPT)
                indices[j] = k
                checkpoint(j,k)
            catch e
                @warn "Maximum bond dimension reached. Skipping to next trotter step" 
                checkpoint(j,k)
                break
            end

        end 
    end

    jldsave(output_path; bond_dimensions, values, trotter, accuracy, indices)
    isfile(checkpoint_path) && rm(checkpoint_path)

    return bond_dimensions, values, indices
end

# find maximum common value 
function maximum_common_χ(bdim, index; ref = bdim[end, index[end]],  tol = 10)
    mcχ_index = zeros(Int, size(bdim,1))
    for t in axes(bdim, 1)
        for k in reverse(1:index[t])
            if isapprox(ref, bdim[t, k], atol = tol)
                mcχ_index[t] = k
                break
            end 
        end
    end
    any(iszero, mcχ_index) && error("Maximum common χ not found with tolerance = $tol. It suggested to lower the reference value and/or increase the tolerance.")
    println("Maximum common χ: $ref ± $tol")
    return ref, mcχ_index
end 
