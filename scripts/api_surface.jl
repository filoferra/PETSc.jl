# ============================================================================
#   High-level API surface extraction
# ============================================================================
#
# Lists every public high-level binding, so the rename table in
# docs/src/man/naming.md can be checked for gaps mechanically rather than by
# hand. Run from the package root:
#
#   julia --project=. scripts/api_surface.jl              # print the surface
#   julia --project=. scripts/api_surface.jl --check      # diff against the docs
#
# Julia has no naming convention for internals: a helper is internal because it
# is not exported and not documented as API, not because of how it is spelled.
# Since only types are exported, the rename table is the register of what is
# public. Every binding must appear there, either with a replacement name or
# marked internal, and this script fails when one appears in neither.
#
# Base and stdlib extensions never appear here: they keep their Base names and
# are governed by §11 of the naming guidelines, not by the rename table.

using PETSc

const DOC = joinpath(@__DIR__, "..", "docs", "src", "man", "naming.md")

safeparent(o) = try
    string(parentmodule(o))
catch
    "?"
end

"""
    surface() -> Vector{NamedTuple}

Every binding in `PETSc` backed by methods defined in PETSc's own source.
"""
function surface()
    rows = NamedTuple[]
    for n in names(PETSc; all = true)
        s = String(n)
        (startswith(s, "#") || s in ("PETSc", "LibPETSc", "eval", "include")) && continue
        isdefined(PETSc, n) || continue
        o = getproperty(PETSc, n)
        o isa Module && continue

        if o isa Type
            push!(rows, (name = s, kind = "type", file = ""))
        elseif o isa Function
            own = filter(m -> startswith(string(m.module), "PETSc"), collect(methods(o)))
            isempty(own) && continue
            file = replace(String(first(own).file), r".*/src/" => "src/")
            push!(rows, (name = s, kind = "function", file = file))
        end
    end
    return sort!(rows, by = r -> (r.file, r.name))
end

"""
    check() -> Int

Report bindings the naming guidelines never mention. Returns the number of
gaps, so CI can fail on a non-zero result.
"""
function check()
    doc = read(DOC, String)
    gaps = [r for r in surface() if !occursin(r.name, doc)]
    if isempty(gaps)
        println("naming.md accounts for every binding.")
    else
        println("bindings missing from naming.md: ", length(gaps))
        for r in gaps
            println("  ", rpad(r.file, 20), r.name)
        end
    end
    return length(gaps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--check" in ARGS
        exit(check() == 0 ? 0 : 1)
    else
        for r in surface()
            println(join((r.name, r.kind, r.file), "\t"))
        end
    end
end
