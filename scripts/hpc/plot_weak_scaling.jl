# ============================================================================
#   Weak-scaling plot of ex45 runs
# ============================================================================
# Plots a time column (default KSPSolve) against the number of ranks for each variant in a
# CSV written by parse_runs.jl: the median as a line, the range of the runs shaded, every
# run as a small marker.
#
# Usage: julia --project=. plot_weak_scaling.jl runs.csv out.png "title" [column]
# column: kspsolve, setup_solve (time to solution), or petsc_time (PetscInitialize to
# PetscFinalize, with Julia's compilation)

using CairoMakie, DelimitedFiles, Statistics

const YLABEL = Dict("kspsolve" => "KSPSolve [s]",
                    "setup_solve" => "setup + solve [s]",
                    "petsc_time" => "PetscInitialize to PetscFinalize [s]")

const STYLE = Dict("native" => (:black, :circle, "native C"),
                   "local" => (:tomato, :diamond, "PETSc.jl (local lib)"),
                   "jll" => (:dodgerblue, :rect, "PETSc.jl (jll)"))

"""
    read_runs(file)

The columns of a CSV written by parse_runs.jl, as a `Dict` from column name to strings.
"""
function read_runs(file)
    cells, header = readdlm(file, ',', String; header = true)
    return Dict(String(h) => cells[:, j] for (j, h) in pairs(vec(header)))
end

"""
    tick_label(runs, n)

Tick label for `n` ranks: ranks, nodes when known, and the global grid.
"""
function tick_label(runs, n)
    i = findfirst(==(string(n)), runs["ranks"])
    nodes = runs["nodes"][i]
    return string(n, isempty(nodes) ? "" : "\n[$nodes]", "\n(", runs["grid"][i], "³)")
end

"""
    plot_weak_scaling(runs, title; column = "kspsolve")

Figure with the median, the range (shaded) and the single-run times of `column` per
variant against ranks.
"""
function plot_weak_scaling(runs, title; column = "kspsolve")
    ranks = parse.(Int, runs["ranks"])
    t = parse.(Float64, runs[column])
    sizes = sort(unique(ranks))
    fig = Figure(size = (800, 500), fontsize = 14)
    ax = Axis(fig[1, 1]; title, xscale = log2, ylabel = get(YLABEL, column, column),
              xlabel = "MPI ranks, [nodes], (global grid)",
              xticks = (sizes, [tick_label(runs, n) for n in sizes]),
              xminorticksvisible = false, yminorticksvisible = false)
    for v in ("native", "local", "jll")
        color, marker, label = STYLE[v]
        sel = runs["variant"] .== v
        any(sel) || continue
        n = sort(unique(ranks[sel]))
        runs_at(m) = t[sel .& (ranks .== m)]
        band!(ax, n, minimum.(runs_at.(n)), maximum.(runs_at.(n)); color = (color, 0.15))
        scatter!(ax, ranks[sel], t[sel]; color = (color, 0.5), marker, markersize = 7)
        med = median.(runs_at.(n))
        scatterlines!(ax, n, med; color, marker, markersize = 12, linewidth = 2, label)
    end
    ylims!(ax, 0, nothing)
    axislegend(ax; position = :lb)
    return fig
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    csv, out, title = ARGS[1:3]
    column = get(ARGS, 4, "kspsolve")
    save(out, plot_weak_scaling(read_runs(csv), title; column); px_per_unit = 2)
end
