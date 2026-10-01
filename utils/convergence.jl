using UniformTEMPO
using JLD2 
using Dates 
using ProgressMeter
using Printf
using InteractiveUtils
using Pkg
using UUIDs

function _atomic_save(dest; kwargs...)
    tmp = dest * ".tmp"
    jldsave(tmp; kwargs...)
    mv(tmp, dest; force = true)
end

"""
    _capture_env()

Snapshot the Julia build, threading configuration, and host machine at the time
of a convergence run.

Fields are captured defensively: an unavailable value is recorded as
`"unavailable"` rather than raising, so provenance capture can never abort a run.
"""
function _capture_env()
    probe(f, default = "unavailable") = try f() catch; default end

    return Dict{String,Any}(
        # --- versions ---
        "julia_version"    => string(VERSION),
        "unitempo_version" => probe(() -> string(pkgversion(UniformTEMPO))),
        "project"          => probe(() -> Base.active_project()),

        # --- threading / BLAS
        "julia_threads"    => Threads.nthreads(),
        "blas_threads"     => probe(() -> BLAS.get_num_threads(), -1),
        "blas_config"      => probe(() -> string(BLAS.get_config())),

        # --- machine ---
        "hostname"         => probe(() -> gethostname()),
        "machine"          => string(Sys.MACHINE),
        "cpu_name"         => probe(() -> Sys.CPU_NAME),
        "cpu_threads"      => Sys.CPU_THREADS,
        "total_memory_gb"  => probe(() -> round(Sys.total_memory() / 2^30; digits = 2), -1.0),

        # --- scheduler / shell ---
        "env_vars"         => _relevant_env_vars(),
    )
end

"""
    _relevant_env_vars()

Capture threading and batch-scheduler environment variables. Only variables that
are actually set are stored, keeping the record compact.
"""
function _relevant_env_vars()
    keys_of_interest = [
        "JULIA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "JULIA_EXCLUSIVE", "JULIA_CPU_TARGET",
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK", "SLURM_ARRAY_TASK_ID",
        "PBS_JOBID", "LSB_JOBID",
    ]
    return Dict{String,String}(k => ENV[k] for k in keys_of_interest if haskey(ENV, k))
end



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
                           checkpoint, pt_path; start_j::Int = 1, start_k::Int = 1, run_metadata)

    n_j, n_k = length(trotter), length(accuracy)
    total = n_j * n_k 
    completed = 0   

    report(j, k, bdim) = update!(prog, completed; showvalues = [
        (:trotter_step,   @sprintf("%.1e (%d/%d)", trotter[j], j, n_j)),
        (:accuracy,       @sprintf("%.1e (%d/%d)", accuracy[k], k, n_k)),
        (:bond_dimension, bdim),
        (:elapsed_s,      round(time() - prog.tinit; digits = 2)),
    ])

    prog = Progress(total; dt = 0.5, desc = "Convergence run: ",
                    barglyphs = BarGlyphs("[=> ]"), color = :cyan)

    for j in start_j:lastindex(trotter)
        k0 = (j == start_j) ? start_k : firstindex(accuracy)
        for k in k0:lastindex(accuracy)
            try
                MyPT = uniTEMPO(S, trotter[j], bcf, accuracy[k]; verbose = false, kwargs...)
                bond_dimensions[j, k] = bond_dim(MyPT)
                values[j, k]          = value_func(MyPT)
                indices[j]            = k

                #saving process tensor and (updated) parameters
                !isnothing(pt_path) && jldsave(joinpath(pt_path, "pt_$(j)_$(k).jld2"); MyPT, trotter = trotter[j], accuracy = accuracy[k], bdim = bond_dimensions[j, k], metadata = run_metadata) 
                !isnothing(pt_path) && jldsave(joinpath(pt_path, "convergence_param.jld2"); trotter, accuracy, bond_dimensions, metadata = run_metadata)

                checkpoint(j, k; broke = false)
                report(j, k, bond_dimensions[j, k])
                completed += 1
               
            catch #the current catch implemention is not very sound. TO DO: differentiate between errors/exceptions
                checkpoint(j, k; broke = true)
                report(j, k, "max bond dimension reached")
                completed += length(k:lastindex(accuracy))
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
    isdir(path) || throw(ArgumentError("Provided path is not a directory: $path"))

    # strip a trailing ".jld2" from filename if present
    endswith(filename, ".jld2") && (filename = filename[1:end-length(".jld2")])

    # assert that path/filename is a directory; if not, create it
    target_dir = joinpath(path, filename)
    !isdir(target_dir) && mkpath(target_dir)


    # if path/filename already contains "filename.jld2" or
    #    "filename.ckpt.jld2", throw an error
    output_path = joinpath(target_dir, filename * ".jld2")
    ckpt_path   = joinpath(target_dir, filename * ".ckpt.jld2")
    isfile(output_path) && error("Convergence results with filename \"$(filename).jld2\" " *"already exist in $target_dir.")

    if isfile(ckpt_path) && !resume 
        error("A checkpoint file \"$(filename).ckpt.jld2\" already exists in " *
            "$target_dir. Use `resume_from_checkpoint()`.")
    end

    if pt_save
        pt_path = joinpath(target_dir, "pt_" * filename)
        if isdir(pt_path)
            resume == false && error("A process-tensor directory \"pt_$(filename)\" already " *"exists in $target_dir. Use `resume_from_checkpoint()` " *"to continue from it.")
        else
            mkpath(pt_path)
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
function _make_checkpoint(checkpoint_path, bond_dimensions, values, trotter, accuracy, indices, run_metadata)
    return (j, k; broke::Bool = false) -> _atomic_save(checkpoint_path; bond_dimensions, values, trotter, accuracy, trotter_index = j, accuracy_index = k, indices, broke, metadata = run_metadata)
end

"""
    convergence(value_func, s, trotter, bcf, accuracy;
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
function convergence(value_func::Function, s::Union{AbstractMatrix{<:Number}, Vector}, trotter::AbstractArray{<:Number}, bcf::Union{Function, Array}, accuracy::AbstractArray{<:Number};
                    path::String = pwd(), filename::String = "convergence", pt_save::Bool = false, 
                    label::String= "", metadata::Dict{String,Any} = Dict{String,Any}(), 
                    kwargs...)

    
   
    # --- Probe cell (1,1): validates inputs, infers T, 
    pt_first = uniTEMPO(s, first(trotter), bcf, first(accuracy); kwargs...)
    v_first  = value_func(pt_first)
    T        = typeof(v_first)
    bdim_first = bond_dim(pt_first)

    # resolve paths
    output_path, checkpoint_path, pt_path = _resolve_paths(path, filename, pt_save)

    # allocate results array
    bond_dimensions = Array{Union{Int64, Missing}}(missing, length(trotter), length(accuracy))
    values = Array{Union{T, Missing}}(missing, length(trotter), length(accuracy))
    indices = Array{Union{Int, Missing}}(missing, length(trotter))

    # save first run 
    bond_dimensions[1] = bdim_first
    values[1] = v_first

    # define convergence run metadata
    run_metadata = merge(Dict{String,Any}(
            "label"      => label,
            "value_type" => string(T),
            "created"    => string(Dates.now()),
            "n_trotter"  => length(trotter),
            "n_accuracy" => length(accuracy),
            "kwargs"  => NamedTuple(kwargs),
            "pt_save" => pt_save,
            "run_id" => string(uuid4()),
            "env" => _capture_env(),
        ), metadata)

    # make first checkpoint
    checkpoint = _make_checkpoint(checkpoint_path, bond_dimensions, values, trotter, accuracy, indices, run_metadata)

    # checkpoint save of first run
    checkpoint(firstindex(trotter), firstindex(accuracy); broke = false)

    # convergence run
    _run_convergence!(value_func, s, trotter, bcf, kwargs, accuracy, bond_dimensions, values, indices, checkpoint, pt_path; start_j = firstindex(trotter), start_k = firstindex(accuracy)+1, run_metadata)

    # save convergence run
    _atomic_save(output_path; bond_dimensions, values, trotter, accuracy, indices, metadata = run_metadata)
    isfile(checkpoint_path) && rm(checkpoint_path)
                
    return bond_dimensions, values, indices
end                    

"""
    resume_from_checkpoint(value_func, s, bcf;
                           path = pwd(), filename = "convergence",
                           label = "", pt_save = false)

Resume an interrupted `convergence` run from its checkpoint file, continuing from
the saved position and finalizing the results. Returns `(bond_dimensions, values, indices)`.

# Arguments
- `value_func`, `s`, `bcf`: re-supplied since they are not stored in the checkpoint.
- `path`, `filename`: locate the checkpoint (keyword).
- `label`: optional label; warns if it differs from the stored one (keyword).
- `pt_save`: whether process tensors are being saved (keyword).
"""
function resume_from_checkpoint(value_func::Function, s::Union{AbstractMatrix{<:Number}, Vector},bcf::Union{Function, Array}; path::String = pwd(), filename::String = "convergence", label::String = "",pt_save::Bool = false)

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
    kwargs          = get(saved_meta, "kwargs", NamedTuple())
    stored_pt       = get(saved_meta, "pt_save", false)

    stored_pt == pt_save || error("pt_save = $pt_save disagrees with checkpoint ($stored_pt)")

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

    # record provenance of this resume. Appending resume history to original saved medatada
    resume_env = _capture_env()

    history = get(saved_meta, "resume_history", Vector{Dict{String,Any}}())
    push!(history, Dict{String,Any}(
        "resumed_at"  => string(Dates.now()),
        "resume_id"   => string(uuid4()),
        "start_index" => (start_j, start_k),
        "env"         => resume_env,
    ))
    saved_meta["resume_history"] = history

    # the one difference that can make the second half of a grid inconsistent
    # with the first: a change in the TEMPO implementation itself
    orig_version   = get(get(saved_meta, "env", Dict{String,Any}()), "unitempo_version", nothing)
    resume_version = get(resume_env, "unitempo_version", nothing)

    if !isnothing(orig_version) && orig_version != resume_version
        @warn "UniformTEMPO version differs from the original run; the resumed " *
              "portion of the grid may not be consistent with the completed part" original=orig_version now=resume_version
    end


    checkpoint = _make_checkpoint(checkpoint_path, bond_dimensions, values,
                                  trotter, accuracy, indices, saved_meta)

    # Already complete: just finalize.
    if start_j > lastindex(trotter)
        @info "Checkpoint already complete; writing final output."
    else
        @info "Resume position" start_trotter = start_j start_accuracy = start_k
        _run_convergence!(value_func, s, trotter, bcf, kwargs, accuracy,
                          bond_dimensions, values, indices,
                          checkpoint, pt_path; start_j = start_j, start_k = start_k, run_metadata = saved_meta)
    end

    
    _atomic_save(output_path; bond_dimensions, values, trotter, accuracy, indices,
            metadata = saved_meta)
    isfile(checkpoint_path) && rm(checkpoint_path)

    return bond_dimensions, values, indices
end


