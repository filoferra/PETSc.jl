# ============================================================================
#   Weak-scaling plot of ex45 runs
# ============================================================================
# Plots KSPSolve time against the number of ranks for each variant in a CSV written by
# parse_runs.jl: the median as a line, every run as a small marker.
#
# Usage: julia --project=. plot_weak_scaling.jl runs.csv out.png "title"

using CairoMakie, DelimitedFiles, Statistics

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
    plot_weak_scaling(runs, title)

Figure with the median and the single-run KSPSolve times per variant against ranks.
"""
function plot_weak_scaling(runs, title)
    ranks = parse.(Int, runs["ranks"])
    ksp = parse.(Float64, runs["kspsolve"])
    sizes = sort(unique(ranks))
    fig = Figure(size = (800, 500), fontsize = 14)
    ax = Axis(fig[1, 1]; title, xscale = log2, ylabel = "KSPSolve [s]",
              xlabel = "MPI ranks, [nodes], (global grid)",
              xticks = (sizes, [tick_label(runs, n) for n in sizes]),
              xminorticksvisible = false, yminorticksvisible = false)
    for v in ("native", "local", "jll")
        color, marker, label = STYLE[v]
        sel = runs["variant"] .== v
        any(sel) || continue
        scatter!(ax, ranks[sel], ksp[sel]; color = (color, 0.5), marker, markersize = 7)
        n = sort(unique(ranks[sel]))
        med = [median(ksp[sel .& (ranks .== m)]) for m in n]
        scatterlines!(ax, n, med; color, marker, markersize = 12, linewidth = 2, label)
    end
    ylims!(ax, 0, nothing)
    axislegend(ax; position = :lb)
    return fig
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    csv, out, title = ARGS
    save(out, plot_weak_scaling(read_runs(csv), title); px_per_unit = 2)
end
