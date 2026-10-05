using UniformTEMPO
using JLD2 
using Dates 
using ProgressMeter
using Printf
using InteractiveUtils
using Pkg
using UUIDs

const PT_DIRNAME = "process_tensors"
const PT_PARAMS_FILE  = "convergence_params.jld2"



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
                !isnothing(pt_path) && jldsave(joinpath(pt_path, PT_PARAMS_FILE); trotter, accuracy, bond_dimensions, metadata = run_metadata)

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
    _resolve_paths(path, model_tag, value_tag, param_tag, pt_save; mode = :fresh)

Validate the output location and return `(output_path, ckpt_path, pt_path)`
without touching the filesystem. The run folder is `path/<model_tag>_<param_tag>`.

Modes:
- `:fresh` (`convergence`): `<value_tag>.jld2` and `<value_tag>.ckpt.jld2` must not
  exist; if `pt_save = true`, the run folder must not exist yet.
- `:resume` (`resume_from_checkpoint`): `<value_tag>.ckpt.jld2` must exist; if
  `pt_save = true`, `process_tensors/` must exist.
- `:from_pts` (`convergence_from_process_tensors`): requires `pt_save = true`;
  `process_tensors/` must exist and contain `PT_PARAMS_FILE` and at least one
  `pt_*.jld2`; `<value_tag>.ckpt.jld2` must not exist.

In every mode, `<value_tag>.jld2` must not exist.
"""
function _resolve_paths(path, model_tag, value_tag, param_tag, pt_save; mode::Symbol = :fresh)
    mode in (:fresh, :resume, :from_pts) || throw(ArgumentError("Unknown mode :$mode"))
    mode == :from_pts && !pt_save && throw(ArgumentError("mode = :from_pts requires pt_save = true"))

    path = abspath(expanduser(path))
    isdir(path) || throw(ArgumentError("Provided path is not a directory: $path"))

    target_dir  = joinpath(path, join(filter(!isempty, [model_tag, param_tag]), "_"))
    output_path = joinpath(target_dir, value_tag * ".jld2")
    ckpt_path   = joinpath(target_dir, value_tag * ".ckpt.jld2")
    pt_path     = pt_save ? joinpath(target_dir, PT_DIRNAME) : nothing

    isfile(output_path) && error("Convergence results \"$(value_tag).jld2\" already exist in $target_dir.")

    if mode == :fresh
        if pt_save && isdir(target_dir)
            error("Run folder $target_dir already exists. Process tensors can only be " *
                  "saved into a new folder: choose a different `model_tag`/`param_tag`, " *
                  "or set `pt_save = false`.")
        end
        isfile(ckpt_path) && error("A checkpoint \"$(value_tag).ckpt.jld2\" already exists in " *
                                   "$target_dir. Use `resume_from_checkpoint()`.")

    elseif mode == :resume
        isfile(ckpt_path) || error("No checkpoint \"$(value_tag).ckpt.jld2\" found in $target_dir.")
        pt_save && !isdir(pt_path) &&
            error("pt_save = true, but no \"$PT_DIRNAME\" directory exists in $target_dir.")

    else  # :from_pts
        isfile(ckpt_path) &&
            error("A checkpoint \"$(value_tag).ckpt.jld2\" exists in $target_dir. Finish that run " *
                  "with `resume_from_checkpoint()` or choose a different `value_tag`.")
        isdir(pt_path) || error("No \"$PT_DIRNAME\" directory found in $target_dir.")
        isfile(joinpath(pt_path, PT_PARAMS_FILE)) || error("No \"$PT_PARAMS_FILE\" found in $pt_path.")
        any(f -> startswith(f, "pt_") && endswith(f, ".jld2"), readdir(pt_path)) ||
            error("The \"$PT_DIRNAME\" directory in $target_dir contains no process tensors.")
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
function convergence(   value_func::Function, 
                        s::Union{AbstractMatrix{<:Number}, Vector}, trotter::AbstractArray{<:Number}, bcf::Union{Function, Array}, accuracy::AbstractArray{<:Number};
                        path::String = pwd(), model_tag::String = "convergence", value_tag::String = "convergence_values", param_tag::String = "",
                        pt_save::Bool = false, metadata::Dict{String,Any} = Dict{String,Any}(), 
                        kwargs...)

    
    # --- Probe cell (1,1): validates inputs, infers T, 
    pt_first = uniTEMPO(s, first(trotter), bcf, first(accuracy); kwargs...)
    v_first  = value_func(pt_first)
    T        = typeof(v_first)
    bdim_first = bond_dim(pt_first)

    # resolve paths
    out_path, ckpt_path, pt_path = _resolve_paths(path, model_tag, value_tag, param_tag, pt_save)
    # probe succeeded: create the run folder (and process_tensors/ if needed)
    mkpath(something(pt_path, dirname(out_path)))

    # allocate results array
    bond_dimensions = Array{Union{Int64, Missing}}(missing, length(trotter), length(accuracy))
    values = Array{Union{T, Missing}}(missing, length(trotter), length(accuracy))
    indices = Array{Union{Int, Missing}}(missing, length(trotter))

    # save first run 
    bond_dimensions[1] = bdim_first
    values[1] = v_first
    indices[1] = 1

    # define convergence run metadata
    run_metadata = merge(Dict{String,Any}(
            "model" => model_tag,
            "value" => value_tag, 
            "params" => param_tag,
            "created" => string(Dates.now()),
            "run_id" => string(uuid4()),
            "env" => _capture_env(),
            
            "value_type" => string(T),
            "pt_save" => pt_save,
            "kwargs"  => NamedTuple(kwargs),
        ), metadata)

    # make first checkpoint
    checkpoint = _make_checkpoint(ckpt_path, bond_dimensions, values, trotter, accuracy, indices, run_metadata)

    # checkpoint save of first run and process tensor save
    checkpoint(firstindex(trotter), firstindex(accuracy); broke = false)
    !isnothing(pt_path) && jldsave(joinpath(pt_path, "pt_$(firstindex(trotter))_$(firstindex(accuracy)).jld2"); MyPT = pt_first, trotter = first(trotter), accuracy = first(accuracy), bdim = first(bond_dimensions), metadata = run_metadata) 
    !isnothing(pt_path) && jldsave(joinpath(pt_path, PT_PARAMS_FILE); trotter, accuracy, bond_dimensions, metadata = run_metadata)
    
    # convergence run
    _run_convergence!(value_func, s, trotter, bcf, kwargs, accuracy, bond_dimensions, values, indices, checkpoint, pt_path; start_j = firstindex(trotter), start_k = firstindex(accuracy)+1, run_metadata)

    # save convergence run
    _atomic_save(out_path; bond_dimensions, values, trotter, accuracy, indices, metadata = run_metadata)
    isfile(ckpt_path) && rm(ckpt_path)
                
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
function resume_from_checkpoint(value_func::Function, s::Union{AbstractMatrix{<:Number}, Vector},bcf::Union{Function, Array}; path::String = pwd(), model_tag::String = "convergence", value_tag::String = "convergence_values", param_tag::String = "",pt_save::Bool = false)

    out_path, ckpt_path, pt_path = _resolve_paths(path, model_tag, value_tag, param_tag, pt_save; mode = :resume)

    # --- load saved state ---
    state           = load(ckpt_path)
    bond_dimensions = state["bond_dimensions"]
    values          = state["values"]
    trotter         = state["trotter"]
    accuracy        = state["accuracy"]
    indices         = state["indices"]
    j_saved         = state["trotter_index"]
    k_saved         = state["accuracy_index"]
    broke           = get(state, "broke", false)
    saved_meta      = get(state, "metadata", Dict{String,Any}())
    saved_value_tag = get(saved_meta, "value", "")
    kwargs          = get(saved_meta, "kwargs", NamedTuple())
    stored_pt       = get(saved_meta, "pt_save", false)

    stored_pt == pt_save || error("pt_save = $pt_save disagrees with checkpoint ($stored_pt)")

    @info "Resuming convergence run" quantity=saved_value_tag created=get(saved_meta, "created", "unknown")

    if saved_value_tag != value_tag
        @warn "Resume value_tag differs from checkpoint" supplied = value_tag stored = saved_value_tag
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


    checkpoint = _make_checkpoint(ckpt_path, bond_dimensions, values,
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

    
    _atomic_save(out_path; bond_dimensions, values, trotter, accuracy, indices,
            metadata = saved_meta)
    isfile(ckpt_path) && rm(ckpt_path)

    return bond_dimensions, values, indices
end



"""
    _load_pt(pt_path, j, k, trotter, accuracy)

Load the process tensor of grid cell `(j, k)` and check that the Trotter step and
accuracy stored alongside it match the grid in `PT_PARAMS_FILE`.
"""
function _load_pt(pt_path, j, k, trotter, accuracy)
    file = joinpath(pt_path, "pt_$(j)_$(k).jld2")
    isfile(file) || error("Process tensor file missing: $file")

    pt, tr, acc = load(file, "MyPT", "trotter", "accuracy")
    (tr ≈ trotter[j] && acc ≈ accuracy[k]) ||
        error("Grid mismatch in $file: stored (trotter = $tr, accuracy = $acc), " *
              "expected (trotter = $(trotter[j]), accuracy = $(accuracy[k])).")
    return pt
end

"""
    convergence_from_process_tensors(value_func;
                                     path = pwd(), model_tag = "convergence",
                                     value_tag = "convergence_values", param_tag = "",
                                     metadata = Dict{String,Any}())

Run a full convergence study of `value_func` using process tensors saved by an
earlier `convergence(...; pt_save = true)` run, without recomputing them.
Returns `(bond_dimensions, values, indices)`.

The grid (`trotter`, `accuracy`) and the bond dimensions are read from
`PT_PARAMS_FILE`. Results are written to `path/<model_tag>_<param_tag>/<value_tag>.jld2`.

# Arguments
- `value_func`: reduces each process tensor to the recorded observable.
- `path`, `model_tag`, `param_tag`: locate the run folder holding the process tensors (keyword).
- `value_tag`: name of the output file for this observable (keyword).
- `metadata`: user annotations merged into the stored metadata (keyword).
"""
function convergence_from_process_tensors(value_func::Function, model_tag::String, param_tag::String, value_tag::String; path::String = pwd(), metadata::Dict{String,Any} = Dict{String,Any}())

    out_path, _, pt_path = _resolve_paths(path, model_tag, value_tag, param_tag, true; mode = :from_pts)
    target_dir = dirname(out_path)
    
    # --- load grid and provenance of the process-tensor run ---
    params    = load(joinpath(pt_path, PT_PARAMS_FILE))
    trotter   = params["trotter"]
    accuracy  = params["accuracy"]
    pt_bdims  = params["bond_dimensions"]
    pt_meta   = get(params, "metadata", Dict{String,Any}())

    # an unfinished PT run leaves its own checkpoint behind: the grid may be partial
    pt_value_tag = get(pt_meta, "value", nothing)
    if !isnothing(pt_value_tag) && isfile(joinpath(target_dir, pt_value_tag * ".ckpt.jld2"))
        @warn "The process-tensor run appears unfinished; the grid may be incomplete" checkpoint = pt_value_tag * ".ckpt.jld2"
    end

    # cells with a saved process tensor, in row-major (trotter, then accuracy) order
    cells = [(j, k) for j in axes(pt_bdims, 1) for k in axes(pt_bdims, 2) if !ismissing(pt_bdims[j, k])]
    isempty(cells) && error("\"$PT_PARAMS_FILE\" in $pt_path records no completed grid cells.")

    # --- probe the first available cell: validates value_func, infers T ---
    j1, k1  = first(cells)
    v_first = value_func(_load_pt(pt_path, j1, k1, trotter, accuracy))
    T       = typeof(v_first)

    # allocate results arrays
    bond_dimensions = Array{Union{Int64, Missing}}(missing, length(trotter), length(accuracy))
    values          = Array{Union{T, Missing}}(missing, length(trotter), length(accuracy))
    indices         = Array{Union{Int, Missing}}(missing, length(trotter))

    # define convergence run metadata
    run_env = _capture_env()
    run_metadata = merge(Dict{String,Any}(
            "model"      => model_tag,
            "value"      => value_tag,
            "params"     => param_tag,
            "created"    => string(Dates.now()),
            "run_id"     => string(uuid4()),
            "env"        => run_env,

            "value_type" => string(T),
            "pt_save"    => false,
            "source"     => "process_tensors",
            "kwargs"     => get(pt_meta, "kwargs", NamedTuple()),
            "pt_run"     => pt_meta,          # full provenance of the run that produced the PTs
        ), metadata)

    # observables of a PT may depend on the package version that reads it
    orig_version = get(get(pt_meta, "env", Dict{String,Any}()), "unitempo_version", nothing)
    now_version  = get(run_env, "unitempo_version", nothing)
    if !isnothing(orig_version) && orig_version != now_version
        @warn "UniformTEMPO version differs from the run that produced the process tensors" original = orig_version now = now_version
    end

    # --- convergence run over saved process tensors ---
    n_j, n_k = length(trotter), length(accuracy)
    prog = Progress(length(cells); dt = 0.5, desc = "Convergence from PTs: ",
                    barglyphs = BarGlyphs("[=> ]"), color = :cyan)

    for (n, (j, k)) in enumerate(cells)
        v = n == 1 ? v_first : value_func(_load_pt(pt_path, j, k, trotter, accuracy))

        bond_dimensions[j, k] = pt_bdims[j, k]
        values[j, k]          = v
        indices[j]            = k

        update!(prog, n; showvalues = [
            (:trotter_step,   @sprintf("%.1e (%d/%d)", trotter[j], j, n_j)),
            (:accuracy,       @sprintf("%.1e (%d/%d)", accuracy[k], k, n_k)),
            (:bond_dimension, bond_dimensions[j, k]),
            (:elapsed_s,      round(time() - prog.tinit; digits = 2)),
        ])
    end
    finish!(prog)

    # save convergence run
    _atomic_save(out_path; bond_dimensions, values, trotter, accuracy, indices, metadata = run_metadata)

    return bond_dimensions, values, indices
end
