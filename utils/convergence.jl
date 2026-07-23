using UniformTEMPO
using JLD2 
using Dates 
using ProgressMeter
using Printf

"""
    _run_convergence!(value_func, S, trotter, bcf, pt_kwargs, accuracy,
                      bond_dimensions, values, indices, checkpoint, pt_path;
                      start_j = 1, start_k = 1)

Internal convergence loop sweeping over `trotter` × `accuracy`.

# Arguments
- `value_func`: reduces a process tensor to the recorded observable.
- `S`, `bcf`: system matrix and bath correlation function passed to `uniTEMPO`.
- `trotter`, `accuracy`: grids of Trotter steps and accuracy values.
- `kwargs`: keyword arguments forwarded to `uniTEMPO`.
- `bond_dimensions`, `values`, `indices`: result containers, mutated in place.
- `checkpoint`: callback `(j, k; broke)` that saves state.
- `pt_path`: directory to save process tensors to file, or `nothing`.
- `start_j`, `start_k`: starting grid indices (for checkpoint resumes).

Returns `(bond_dimensions, values, indices)`.
"""
function _run_convergence!(value_func, S, trotter, bcf, kwargs, accuracy,
                           bond_dimensions, values, indices,
                           checkpoint, pt_path; start_j::Int = 1, start_k::Int = 1)

    total = length(trotter) * length(accuracy)
    n_j   = length(trotter)     # max index for the trotter loop
    n_k   = length(accuracy)    # max index for the accuracy loop

    prog = Progress(total; dt = 0.5, desc = "Convergence run: ",
                    barglyphs = BarGlyphs("[=> ]"), color = :cyan)

    completed = 0

    for j in start_j:lastindex(trotter)
        k0 = (j == start_j) ? start_k : firstindex(accuracy)
        for k in k0:lastindex(accuracy)
            try
                MyPT = uniTEMPO(S, trotter[j], bcf, accuracy[k]; verbose = false, kwargs...)
                bond_dimensions[j, k] = bond_dim(MyPT)
                values[j, k]          = value_func(MyPT)
                indices[j]            = k
                checkpoint(j, k; broke = false)

                completed += 1
                update!(prog, completed;
                        showvalues = [
                            (:trotter_step,   @sprintf("%.1e (%d/%d)", trotter[j], j, n_j)),
                            (:accuracy,       @sprintf("%.1e (%d/%d)", accuracy[k], k, n_k)),
                            (:bond_dimension, bond_dimensions[j, k]),
                            (:elapsed_s,      round(time() - prog.tinit; digits = 2)),
                        ])

                #saving process tensor 
                !isnothing(pt_path) && jldsave(joinpath(pt_path, "pt_$(j)_$(k).jld2"); MyPT, trotter = trotter[j], accuracy = accuracy[j], bdim = bond_dimensions[j, k])

            catch 
                # uncheck for explicit warning
                # @warn "Maximum bond dimension reached. Skipping to next trotter step"
                checkpoint(j, k; broke = true)
                completed += lastindex(accuracy) - k + 1
                
                update!(prog, completed;
                        showvalues = [
                            (:trotter_step,   @sprintf("%.1e (%d/%d)", trotter[j], j, n_j)),
                            (:accuracy,       @sprintf("%.1e (%d/%d)", accuracy[k], k, n_k)),
                            (:bond_dimension, "max_bond dimension reached"),
                            (:elapsed_s,      round(time() - prog.tinit; digits = 2)),
                        ])
                break
            end
        end
    end

    finish!(prog)
    return bond_dimensions, values, indices
end

"""
    _resolve_paths(path, filename, pt_save; resume = false)

Validate `path`, create the `path/filename` output directory, and return the tuple
`(output_path, ckpt_path, pt_path)`. Errors on pre-existing results/checkpoint/PT
artifacts unless `resume` is `true`.

# Arguments
- `path`: existing base directory.
- `filename`: run name (trailing `.jld2` is stripped).
- `pt_save`: if `true`, prepare a `pt_<filename>` directory; else `pt_path = nothing`.
- `resume`: allow existing checkpoint/PT files.
"""
function _resolve_paths(path, filename, pt_save; resume::Bool = false)
    # assert that `path` points to a directory
    path = abspath(expanduser(path))
    @assert isdir(path) "Provided path is not a directory: $path"

    # strip a trailing ".jld2" from filename if present
    endswith(filename, ".jld2") && (filename = filename[1:end-length(".jld2")])

    # assert that path/filename is a directory; if not, create it
    target_dir = joinpath(path, filename)
    !isdir(target_dir) && mkdir(target_dir)


    # if path/filename already contains "filename.jld2" or
    #    "filename.ckpt.jld2", throw an error
    output_path = joinpath(target_dir, filename * ".jld2")
    ckpt_path   = joinpath(target_dir, filename * ".ckpt.jld2")
    isfile(output_path) && error("Convergence results with filename \"$(filename).jld2\" " *"already exist in $target_dir.")

    if (isfile(ckpt_path) && resume == false)
        error("A checkpoint file \"$(filename).ckpt.jld2\" already exists in " *
            "$target_dir. Use `resume_from_checkpoint()` to continue from it.")
    end

    if pt_save
        pt_path = joinpath(target_dir, "pt_" * filename)
        if isdir(pt_path)
            resume == false && error("A process-tensor directory \"pt_$(filename)\" already " *"exists in $target_dir. Use `resume_from_checkpoint()` " *"to continue from it.")
        else
            mkdir(pt_path)
        end
    else
        pt_path = nothing
    end

    return output_path, ckpt_path, pt_path
end

"""
    _make_checkpoint(checkpoint_path, bond_dimensions, values, trotter,
                     accuracy, indices, run_metadata)

Return a closure `(j, k; broke = false)` that atomically writes the current run
state to `checkpoint_path` (via a temp file + `mv`).

# Arguments
- `checkpoint_path`: destination checkpoint file.
- `bond_dimensions`, `values`, `indices`: current result arrays.
- `trotter`, `accuracy`: parameter grids.
- `run_metadata`: metadata dictionary to embed.
"""

function _make_checkpoint(checkpoint_path, bond_dimensions, values,
                          trotter, accuracy, indices, run_metadata)
    return (j, k; broke::Bool = false) -> begin
        tmpfile = checkpoint_path * ".tmp"
        jldsave(tmpfile; bond_dimensions, values, trotter, accuracy,
                trotter_index = j, accuracy_index = k, indices,
                broke, metadata = run_metadata)
        mv(tmpfile, checkpoint_path; force = true)
    end
end

"""
    convergence(value_func, S, trotter, bcf, accuracy;
                path = pwd(), filename = "convergence", pt_save = false,
                label = "", metadata = Dict{String,Any}(), kwargs...)

Run a full convergence study of `value_func` over the `trotter` × `accuracy` grid,
checkpointing throughout and writing final results to disk. Returns
`(bond_dimensions, values, indices)`.

# Arguments
- `value_func`: reduces each process tensor to the recorded observable.
- `S`, `bcf`: system matrix and bath correlation function.
- `trotter`, `accuracy`: grids of Trotter steps and accuracy targets.
- `path`, `filename`: output location (keyword).
- `pt_save`: also serialize individual process tensors (keyword).
- `label`, `metadata`: user annotations stored in metadata (keyword).
- `kwargs...`: forwarded to `uniTEMPO`.
"""

function convergence(value_func::Function, S::AbstractMatrix{<:Number}, trotter::AbstractArray{<:Number}, bcf::Function,accuracy::AbstractArray{<:Number};
                    path::String = pwd(), filename::String = "convergence", pt_save::Bool = false, 
                    label::String= "", metadata::Dict{String,Any} = Dict{String,Any}(), 
                    kwargs...)

    # resolve paths
    output_path, checkpoint_path, pt_path = _resolve_paths(path, filename, pt_save)


    # probe return type (forward the same kwargs for consistency).
    pt = uniTEMPO(S, trotter[1], bcf, accuracy[1]; kwargs...)
    T  = typeof(value_func(pt))

    # allocate results array
    bond_dimensions = Array{Union{Int64, Nothing}}(nothing, length(trotter), length(accuracy))
    values          = Array{Union{T, Nothing}}(nothing, length(trotter), length(accuracy))
    indices     = Array{Int}(undef, length(trotter))

    # define convergence run metadata
    run_metadata = merge(Dict{String,Any}(
            "label"      => label,
            "value_type" => string(T),
            "created"    => string(Dates.now()),
            "n_trotter"  => length(trotter),
            "n_accuracy" => length(accuracy),
            "kwargs"  => NamedTuple(kwargs),
        ), metadata)

    # make first checkpoint
    checkpoint = _make_checkpoint(checkpoint_path, bond_dimensions, values, trotter, accuracy, indices, run_metadata)

    # convergence run
    _run_convergence!(value_func, S, trotter, bcf, kwargs, accuracy, bond_dimensions, values, indices, checkpoint, pt_path)

    # save convergence run
    jldsave(output_path; bond_dimensions, values, trotter, accuracy, indices, metadata = run_metadata)
    isfile(checkpoint_path) && rm(checkpoint_path)
                
    return bond_dimensions, values, indices
end                    

"""
    resume_from_checkpoint(value_func, S, bcf;
                           path = pwd(), filename = "convergence",
                           label = "", pt_save = false)

Resume an interrupted `convergence` run from its checkpoint file, continuing from
the saved position and finalizing the results. Returns `(bond_dimensions, values, indices)`.

# Arguments
- `value_func`, `S`, `bcf`: re-supplied since they are not stored in the checkpoint.
- `path`, `filename`: locate the checkpoint (keyword).
- `label`: optional label; warns if it differs from the stored one (keyword).
- `pt_save`: whether process tensors are being saved (keyword).
"""

function resume_from_checkpoint(value_func::Function, S::AbstractMatrix{<:Number},
                                bcf::Function;
                                path::String = pwd(),
                                filename::String = "convergence",
                                label::String = "",
                                pt_save::Bool = false)

    output_path, checkpoint_path, pt_path = _resolve_paths(path, filename, pt_save; resume = true)
    @assert isfile(checkpoint_path) "No checkpoint found at: '$checkpoint_path'"

    # --- load saved state ---
    state           = load(checkpoint_path)
    bond_dimensions = state["bond_dimensions"]
    values          = state["values"]
    trotter         = state["trotter"]
    accuracy        = state["accuracy"]
    indices         = state["indices"]
    j_saved         = state["trotter_index"]
    k_saved         = state["accuracy_index"]
    broke           = get(state, "broke", false)
    saved_meta      = get(state, "metadata", Dict{String,Any}())
    saved_label     = get(saved_meta, "label", "")
    kwargs          = saved_meta["kwargs"]

    @info "Resuming convergence run" quantity=saved_label value_type=get(saved_meta, "value_type", "unknown") created=get(saved_meta, "created", "unknown")

    if !isempty(label) && !isempty(saved_label) && label != saved_label
        @warn "Resume label differs from checkpoint" supplied = label stored = saved_label
    end


    # --- determine restart position ---
    # (j_saved, k_saved) was already attempted; `broke` means the row was abandoned.
    if broke || k_saved == lastindex(accuracy)
        start_j = j_saved + 1
        start_k = firstindex(accuracy)
    else
        start_j = j_saved
        start_k = k_saved + 1
    end

    checkpoint = _make_checkpoint(checkpoint_path, bond_dimensions, values,
                                  trotter, accuracy, indices, saved_meta)

    # Already complete: just finalize.
    if start_j > lastindex(trotter)
        @info "Checkpoint already complete; writing final output."
    else
        @info "Resume position" start_trotter = start_j start_accuracy = start_k
        _run_convergence!(value_func, S, trotter, bcf, kwargs, accuracy,
                          bond_dimensions, values, indices,
                          checkpoint, pt_path; start_j = start_j, start_k = start_k)
    end

    jldsave(output_path; bond_dimensions, values, trotter, accuracy, indices,
            metadata = saved_meta)
    isfile(checkpoint_path) && rm(checkpoint_path)

    return bond_dimensions, values, indices
end


