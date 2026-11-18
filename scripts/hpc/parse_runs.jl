# ============================================================================
#   Parse ex45 runs
# ============================================================================
# One CSV row per run of examples/dmda/ex45.jl or ex45_julia.c, read from its output
# with -log_view. The variant comes from the log: the C program names itself
# ex45_julia, and PETSc_jll reports Yggdrasil's install prefix in its configure options.
#
# Usage: julia parse_runs.jl logs... > runs.csv

const COLUMNS = ["file", "variant", "ranks", "nodes", "grid", "levels", "smoother", "coarse",
                 "iterations", "L2", "kspsolve", "pcsetup", "petsc_time", "setup", "solve",
                 "aypx_ratio", "setup_solve"]

"""
    first_capture(rx, text)

The first capture group of `rx` in `text`, or `""` when `rx` doesn't match.
"""
first_capture(rx, text) = (m = match(rx, text); m === nothing ? "" : String(m[1]))

"""
    log_stage(text, name)

The event table of the `-log_view` stage called `name`, or `""` when the log has none.
"""
function log_stage(text, name)
    parts = split(text, "--- Event Stage $name")
    length(parts) < 2 && return ""
    return first(split(parts[2], "--- Event Stage"))
end

"""
    event_column(stage, event, column)

Column `column` (1 = count) of `event` in a `-log_view` event table: 3 is the time of the
slowest rank [s], 4 the ratio of slowest to fastest rank.
"""
function event_column(stage, event, column)
    for line in eachline(IOBuffer(stage))
        w = split(line)
        length(w) > column && w[1] == event && return String(w[column + 1])
    end
    return ""
end

"""
    variant(text)

`"native"` for the C program, `"jll"` for PETSc_jll and `"local"` for any other library.
"""
function variant(text)
    occursin(r"ex45_julia on a", text) && return "native"
    occursin("--prefix=/workspace/destdir", text) && return "jll"
    return "local"
end

"""
    coarse_solve(text)

Where the coarse LU runs: on the telescoped ranks, or on all ranks when no reduction
option took effect. PETSc ignores a misspelled `-mg_coarse_telescope_reduction_factor`.
"""
coarse_solve(text) = occursin("-mg_coarse_pc_telescope_reduction_factor", text) ? "reduced" : "all ranks"

"""
    setup_and_solve(setup, solve)

Time for assembling the operators, the multigrid setup and the solve [s]. ex45.jl
times setup and solve separately; ex45_julia.c prints no setup time, and its solve
time includes the setup, which KSPSolve does first.
"""
setup_and_solve(setup, solve) =
    isempty(setup) ? solve : string(round(parse(Float64, setup) + parse(Float64, solve); digits = 6))

"""
    parse_run(file)

The CSV fields of one run, in the order of `COLUMNS`, or `nothing` when `file` has no
`-log_view` summary (a run that failed or was cut off).
"""
function parse_run(file)
    text = read(file, String)
    ranks = first_capture(r"with (\d+) process(?:es)?, by", text)
    isempty(ranks) && return nothing
    main = log_stage(text, "0: Main Stage")
    setup = first_capture(r"Setup time:\s+(\S+)", text)
    solve = first_capture(r"^Solve time:\s+(\S+)"m, text)
    return [basename(file), variant(text), ranks,
            first_capture(r"job \d+: (\d+) nodes", text),   # written by daint/run.sh
            first_capture(r"-Nx (\d+)", text),
            first_capture(r"-pc_mg_levels (\d+)", text),
            first_capture(r"-mg_levels_pc_type (\S+)", text),
            coarse_solve(text),
            first_capture(r"iterations (\d+)\s*$"m, text),
            first_capture(r"L2 error:\s+(\S+)", text),
            event_column(main, "KSPSolve", 3),
            event_column(main, "PCSetUp", 3),
            first_capture(r"^Time \(sec\):\s+(\S+)"m, text),
            setup,
            solve,
            # VecAYPX doesn't communicate, so a ratio well above 1.2 means some cores ran slow
            event_column(log_stage(text, "1: MG Apply"), "VecAYPX", 4),
            setup_and_solve(setup, solve)]
end

"""
    write_runs(io, files)

Write the header and one CSV row per parsed file to `io`; files without a `-log_view`
summary are reported on stderr and left out.
"""
function write_runs(io, files)
    println(io, join(COLUMNS, ","))
    for file in files
        row = parse_run(file)
        if row === nothing
            @warn "no -log_view summary, skipped" file
        else
            println(io, join(row, ","))
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && write_runs(stdout, ARGS)
