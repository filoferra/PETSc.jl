# Naming and API conventions

These rules govern the **high-level interface**: everything reachable as `PETSc.foo`.
The low-level layer (`LibPETSc.*`) mirrors the PETSc C API name for name and is out of scope; it keeps its C names so that PETSc's own documentation stays usable.

The conventions take effect in v0.5 and break the existing API. Renamed functions keep a deprecation shim for one minor cycle. 
A few changes are semantic and cannot be shimmed; those are listed in [§16](#16.-Breaking-changes-without-a-shim).

## 1. Scope and the tiebreak

| Layer | Namespace | Naming |
|---|---|---|
| Low level | `PETSc.LibPETSc` | Verbatim PETSc C names (`DMStagGetCorners`) |
| High level | `PETSc` | These conventions |

Every function outside `LibPETSc` follows these rules, including thin wrappers around a single C call.

Two goals pull against each other here: being idiomatic Julia, and staying recognisable to someone reading the PETSc manual. When they conflict:

> **Julia idiom decides structure. PETSc decides vocabulary.**

Dispatch, `!`, argument order, and extending `Base` follow Julia. Which *word* to use follows PETSc. 
So `DMDAGetDof` becomes `ndofs(dm)`: Julian in shape, PETSc's term for the thing.

### 1.1 What these rules cover

The high-level surface is currently 139 functions and 12 types, obtained from `names(PETSc; all = true)` filtered to bindings whose methods are defined in PETSc's own source. 
Regenerate it with `scripts/api_surface.jl` when auditing the rename table.

Three subsets are treated differently:

**Internal helpers** are exempt: they may be renamed freely, with no shim.

A leading underscore stays legitimate for one thing: distinguishing an inner worker from the wrapper that shares its name, as `Base` does with `_growend!`. As of now, `_mul!` and `_unsafe_localarray` are the two cases in this package. 
That is disambiguation, not a visibility marker.

Because [§13](#13.-Exports) exports only types, "unexported" cannot by itself separate the public API from internals. 
The rename table is the register: every binding is listed either with a replacement name or as internal. `scripts/api_surface.jl --check` fails if a binding appears in neither.

**Macros** follow the same rules as functions: snake_case, and no prefix that repeats the module name. `PETSc.@petsc_residual_fn` says "petsc" twice.

```julia
@residual_fn      # was @petsc_residual_fn
@jacobian_fn      # was @petsc_jacobian_fn
@bd_fn            # was @petsc_bd_fn
@simple_fn        # was @petsc_simple_fn
```

**Base extensions** are not renamed. `Base.size`, `Base.getindex` and the rest keep their Base names by definition; [§11](#11.-Extending-Base-and-stdlib) governs which ones may exist at all.

## 2. Case

Default to `snake_case` for any multi-word name. Run words together only for names on
the closed list below.

```julia
set_type!(ksp, :gmres)
ghost_corners(dm)
ownership_range(A)
local_indices(dm)
local_coordinate_array(dm)
project_function!(vec, dm, f)
```

**Closed run-together list**:

- Base-style predicates: `issimplex`, `issymmetric`, `ishermitian`, `isassembled`,
  `isinitialized`, `isfinalized`
- PETSc single tokens: `ndofs`, `nextra`, `l2diff`, `nullspace`, `seqaij`, `seqdense`
- Type accessors mirroring `Base.eltype`: `scalartype`, `inttype`, `realtype`

The default requires no judgement, and the exceptions are countable. That matters more than matching `Base`, which is itself split: `setindex!` and `set_zero_subnormals` both ship in `Base`, and the Julia style guide only asks for underscores "as necessary to improve readability".

Never use `camelCase`. `PascalCase` means "this is a type", without exception
(see [§5](#5.-Types)).

## 3. Accessors and setters

Readers are nouns. Drop PETSc's `Get`:

```julia
corners(dm)      # was getcorners
info(dm)         # was getinfo
dm(ksp)          # was getDM
solution(ksp)    # was get_solution
type(ksp)        # was gettype
comm(v)          # was getcomm
```

Writers are `set_*!`:

```julia
set_dm!(snes, dm)          # was setDM!
set_type!(ksp, :gmres)     # was KSPSetType
set_values!(A, …)          # was setvalues!
set_name!(obj, name)       # was petsc_setname!
```

This targets PETSc's C `Get`/`Set` prefixes. Base's `get(collection, key, default)` idiom is unaffected, and a function following it may keep `get` in its name.

### 3.1 What accessors return and setters take

PETSc has two kinds of named value, and they are represented differently.

**Enumerations** are fixed sets, and the wrapped layer already exposes 187 of them as
Julia enums. Use the enum:

```julia
DMStag(petsclib, comm, (DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE), …)
```

**Type names** (`KSPType`, `MatType`, `DMType`, `PCType`, …) are not enums and cannot become ones: PETSc registers them at runtime as strings, so plugins add new ones the package has never heard of. 
They are `Symbol` at the Julia API and `String` at the C boundary:

```julia
set_type!(ksp, :gmres)
type(ksp)                                  # :gmres, not "gmres"
PetscMat(petsclib, m, n; type = :seqaij)
DMStag(…; stencil_type = :box)
```

Symbol costs nothing to compare, needs no fixed set (so plugin types keep working), and matches `parse_options`, which already returns a `NamedTuple` with Symbol keys. 
The conversion to `String` happens once, where the call meets C.

A note for contributors: `dm` is both an accessor and the conventional variable name for a DM. 
Inside the package, `dm = dm(ksp)` shadows the function and breaks every later call in that scope. 
Users are unaffected, because [§13](#13.-Exports) does not export `dm` and a local binding cannot shadow a qualified `PETSc.dm`.

## 4. Object prefixes

Drop `dm_`, `mat_`, `vec_`, `plex_` when the first argument already carries the type.
Dispatch does the work:

```julia
project_function!(vec, dm, f)   # was dm_project_function!
set_nullspace!(A, ns)           # was mat_set_null_space!
distribute!(dm)                 # was plexdistribute!
```

Keep a prefix only for free functions, where no argument identifies the domain. 
These take `petsclib` and `comm` rather than a PETSc object:

```julia
fe_create_default(petsclib, comm, dim, Nc, …)
fe_create_lagrange(petsclib, comm, dim, Nc, …)
mat_nullspace_create(petsclib, comm; has_const = true)
```

Check this against the argument list, not against the name. `vtk_save!` looks like a free function and is not one: it takes a `PetscVec` ([§8](#8.-Argument-order)).

## 5. Types

### 5.1 One meaning for PascalCase

Before v0.5, every high-level constructor was a PascalCase *function* whose name differed from the *struct* it returned:

```
KSP()      -> PetscKSP      VecSeq()    -> PetscVec
SNES()     -> PetscSNES     MatSeqAIJ() -> PetscMat
Options()  -> PetscOptions  DMDA()      -> PetscDM
DMStag()   -> PetscDM       DMPlex()    -> PetscDM
```

PascalCase meant either a type or a factory returning something else, and the two never agreed. 
`DMStag()` returning a plain `PetscDM` is the direct cause of the string-based dispatch described in [§5.3](#5.3-DM-flavour-is-a-type,-not-a-string).

The rule is now:

> **PascalCase always names a type. Construction always goes through the type.**

Most of what follows is a consequence of that one sentence.

### 5.2 Prefixes

Use PETSc's bare class name. Add a `Petsc` prefix only when the bare name is too generic to export safely:

```julia
# prefixed
PetscVec  PetscMat  PetscOptions  PetscDS
PetscDM   # low-level handle only, see §5.4

# bare
DMDA  DMStag  DMPlex
KSP  SNES  TS  Tao  IS  AO  PF
```

This renames `PetscKSP` to `KSP` and `PetscSNES` to `SNES`, which also removes their constructor/struct name mismatch. `Vec` and `Mat` stay prefixed: they are exported, and bare `Vec`/`Mat` would clash across the ecosystem.

### 5.3 DM flavour is a type, not a string

Before v0.5 there was a single concrete `PetscDM{PetscLib}`, and flavour was resolved at runtime by comparing a string returned from C:

```julia
# v0.4: type-unstable, the two branches return different NamedTuples
function getcorners(dm)
    type = gettype(dm)
    if type == "da";       return getcorners_dmda(dm)
    elseif type == "stag"; return getcorners_dmstag(dm)
    end
end
```

That forced the `_dmda`/`_dmstag` suffixes and made the return type unpredictable. 
In v0.5 flavour and dimension are type parameters, so ordinary dispatch applies and each method has one concrete return type:

```julia
abstract type AbstractPetscDM{PetscLib} end

DMDA{PetscLib,N}   <: AbstractPetscDM{PetscLib}
DMStag{PetscLib,N} <: AbstractPetscDM{PetscLib}
DMPlex{PetscLib,N} <: AbstractPetscDM{PetscLib}

corners(dm::DMDA{L,N})   where {L,N} = …   # (lower, upper, size)
corners(dm::DMStag{L,N}) where {L,N} = …   # (lower, upper, size, nextra)
```

There are therefore **no type suffixes** on function names. `getcorners_dmstag` becomes a method of `corners`, not a separate function.

Flavour becomes a type parameter only when a high-level method must dispatch on it. 
DM qualifies, because `corners`, `local_indices` and coordinate handling all differ by flavour. 
Vec and Mat do not: their flavour only affects construction, so it stays a keyword ([§9](#9.-Keyword-versus-positional)).

### 5.4 DMs of unknown provenance

`PetscDM{PetscLib}` remains as the low-level handle, since `LibPETSc.DMCreate` and friends must return something before the flavour is known. It is not part of the high-level API.

A DM obtained from another object (`dm(ksp)`, a `DMPlex` read from file) has its flavour and dimension known only at runtime. Such boundary functions query `DMGetType` and `DMGetDimension` and return the concrete type:

```julia
d = dm(ksp)   # ::Union{DMDA{L,1},…,DMStag{L,3},…}
```

That Union has more than nine members, well past `MAX_UNION_SPLITTING`, so inference gives up and the call is a dynamic dispatch. 
One dispatch is cheap. What is not cheap is propagating an abstractly-typed DM through a hot loop, so narrow once behind a function barrier:

```julia
function work(ksp)
    d = dm(ksp)     # dynamic dispatch here, once
    _work(d)        # specialized on the concrete type
end
```

### 5.5 Abstract, callback and wrapper types

**Abstract types** are `Abstract` followed by whatever [§5.2](#5.2-Prefixes) gives the concrete name, so the prefix decision is made once:

```julia
AbstractPetscVec  AbstractPetscMat  AbstractPetscDM      # concrete is prefixed
AbstractKSP  AbstractSNES  AbstractTS  AbstractIS  AbstractAO
```

This renames `AbstractPetscKSP` and `AbstractPetscSNES`, matching `PetscKSP` becoming `KSP`. 
It also fixes `AbstractPETScMemBackend`, the one type spelling the prefix `PETSc` where every other type spells it `Petsc`. 
That one is exported, so the inconsistency is user-visible.

**Callback types** take a `Fn` suffix, following the autowrapped layer's existing `DMDATSRHSFunctionLocalFn`. The current `Fn_` prefix puts an underscore inside a PascalCase name, which no rule here permits:

```julia
KSPComputeRHSFn        # was Fn_KSPComputeRHS
KSPComputeOperatorsFn  # was Fn_KSPComputeOperators
SNESSetFunctionFn      # was Fn_SNESSetFunction
SNESSetJacobianFn      # was Fn_SNESSetJacobian
```

**Wrapper and alias types** keep bare PETSc names. [§5.2](#5.2-Prefixes) asks whether *this name* is too generic to export, not whether its class is: `Mat` needs the prefix, `MatShell` does not.

```julia
MatShell  MatOp  MatPtr  VecPtr        # unchanged
MatOrTranspose                          # was MatAT
```

`MatAT` is a `Union{PetscMat, Transpose, Adjoint}` alias whose name does not say so.

## 6. Constructors

Construction goes through the type, dispatching on argument types. 
This replaces the `MatXxx`/`VecXxx` factory family, and removes the confusion between `MatCreateSeqAIJ` (converted a `SparseMatrixCSC`) and `MatSeqAIJ` (allocated from sizes), which were unrelated functions with near-identical names:

```julia
A = PetscMat(petsclib, S::SparseMatrixCSC)        # was MatCreateSeqAIJ
A = PetscMat(petsclib, m, n, nnz)                 # was MatSeqAIJ
A = PetscMat(petsclib, m, n; type = :dense)       # was MatSeqDense
A = PetscMat(petsclib, rowptr, colval, nzval)     # was MatSeqAIJWithArrays

v = PetscVec(petsclib, 10)                        # was VecSeq
v = PetscVec(petsclib, x::Vector)                 # was VecSeq
```

`petsclib` appears in constructors because no object exists yet to carry it. 
It appears nowhere else ([§8](#8.-Argument-order)).

## 7. Mutation

A function gets `!` if and only if it mutates one of its arguments, **or** package-global state. 
Configuring an opaque PETSc object counts as mutation:

```julia
corners(dm)              # reads
solution(ksp)            # reads
ndofs(dm)                # reads

destroy!(dm)             # was destroy
setup!(A)
assemble!(A)
ghost_update!(v)
set_type!(ksp, :gmres)
solve!(x, ksp, b)

set_library!(path)       # mutates package state, no argument mutated
unset_library!()
```

The global-state clause exists so that `set_library!` and `unset_library!` keep their bang, following `Random.seed!`. 
It does not apply to `set_petsclib`, which despite its name mutates nothing: it builds and returns a library handle, and becomes a `PetscLibType` constructor ([§6](#6.-Constructors)).

`destroy` becomes `destroy!`, and finalizer registrations change with it (`finalizer(destroy!, v)`).

## 8. Argument order

Subject first, and mutated arguments before read-only ones, matching `mul!(C, A, B)` and `copyto!(dest, src)`:

```julia
solve!(x, ksp, b)                  # x is written
project_function!(vec, dm, f)      # vec is written
global_to_local!(lvec, dm, gvec)   # lvec is written

corners(dm)                        # readers: subject first
assemble!(A)
```

`petsclib` never leads a high-level call. The object carries `PetscLib` as a type parameter, so passing it again is redundant. 
The exceptions are constructors ([§6](#6.-Constructors)) and the free functions of [§4](#4.-Object-prefixes).

A sweep of v0.4 found seven functions taking `petsclib` first while also taking a dispatchable PETSc object, so the first argument is recoverable from the second and is dropped:

```
add_boundary!       add_natural_boundary!   dm_compute_l2diff
dm_project_field!   dm_project_function!    plex_set_snes_local_fem!
vtk_save!
```

`vtk_save!` is in that list, which corrects [§4](#4.-Object-prefixes): it takes a `PetscVec`, so it is not a free function and does not keep its prefix. 
It becomes `save_vtk!(vec, filename)`, with `comm` recovered from the vector. 
`vtk_save_fields!` takes an iterable of vectors and becomes `save_vtk!(vecs, filename)` on the same name.

### 8.1 Callbacks come first

A function-valued argument goes first, ahead of the subject, so `do` syntax works. 
This outranks subject-first, and follows `map(f, c)` and `open(f, path)`:

```julia
set_function!(snes, x) do f, x
    …
end

with_local_array!(v) do arr
    arr .= 1
end
```

PETSc callbacks are where `do` earns its keep, and `setfunction!`, `setjacobian!` and `withlocalarray!` already take the callback first in v0.4.

## 9. Keyword versus positional

Positional arguments are the ones the object cannot exist without. 
Keyword arguments are the ones with a sensible default:

```julia
DMStag(petsclib, comm, boundary, dims, dof;
       stencil_width = 1, stencil_type = :box)

PetscMat(petsclib, m, n, nnz)             # required
PetscMat(petsclib, m, n; type = :dense)   # optional, defaults to :seqaij

distribute!(dm; overlap = 0)
```

Never take a keyword argument that changes the return type. `type = :dense` is acceptable because every variant returns `PetscMat`; a keyword selecting a DM flavour would not be, which is why flavour is a type parameter there ([§5.3](#5.3-DM-flavour-is-a-type,-not-a-string)).

## 10. Abbreviations

Follow PETSc's own abbreviations, so a user reading the PETSc manual guesses the Julia name correctly:

```julia
ndofs(dm)        # DMDAGetDof
l2diff(dm, …)    # DMComputeL2Diff
nextra           # DMStagGetCorners
seqaij, dense, mpi, ds, fe, is, pc, ksp, snes, ts
```

Do not invent abbreviations PETSc does not use, and do not expand ones it does.

## 11. Extending Base and stdlib

Add methods to `Base` and `LinearAlgebra` wherever the PETSc operation satisfies the existing contract, so that generic Julia code works on PETSc objects:

```julia
Base.size, Base.length, Base.ndims, Base.eltype, Base.axes
Base.similar, Base.fill!, Base.iterate, Base.copyto!
Base.getindex, Base.setindex!, Base.show
LinearAlgebra.norm, LinearAlgebra.mul!,
LinearAlgebra.issymmetric, LinearAlgebra.ishermitian
```

### 11.1 Blocklist

Do **not** extend these. The word matches but the contract does not, and a silent mismatch is worse than an unfamiliar name:

| Name | Why not | Use instead |
|---|---|---|
| `LinearAlgebra.nullspace` | Returns a basis matrix; PETSc's `MatNullSpace` is a solver hint object | `set_nullspace!`, `mat_nullspace_create` |
| `Base.values` | PETSc "values" are a setter concept (`MatSetValues`), not a view of contents | `entries` |
| `Base.view` | `PetscViewer` writes an object to a stream; it is not an array view | `petscview`, or `Base.show` |
| `Base.setfield!` | Core builtin that writes a struct field; PETSc's attaches a finite element to a DM | `set_field!` |

Additions need the same three columns: the name, the contract mismatch, and the replacement.

`setfield!` is worth a note. v0.4 defines `PETSc.setfield!` with 8 methods as a function distinct from the core builtin, so inside the module the builtin is shadowed. 
§2's snake_case rule resolves it without a special case: `set_field!` and `setfield!` are different identifiers.

## 12. Return values

Functions returning several related values return a `NamedTuple`, with field names from a fixed vocabulary:

| Field | Type | Meaning |
|---|---|---|
| `lower`, `upper` | `CartesianIndex{N}` | 1-based, inclusive |
| `size` | `NTuple{N,Int}` | Local extent |
| `center`, `vertex` | `NamedTuple` of `UnitRange` | Staggered index ranges, keyed `x`, `y`, `z` |
| `x`, `y`, `z` | `UnitRange` | Per-axis range inside `center`/`vertex` |
| `nextra` | `NTuple{N,Int}` | DMStag partial elements |

```julia
c = corners(dm2d)
c.lower                      # CartesianIndex{2}
(; lower, upper) = corners(dm)
```

Extend the vocabulary rather than inventing a synonym: a local extent is `size`, never `dims` or `extent`.

A sweep of the high-level layer found six functions returning a `NamedTuple` (`corners` and `ghost_corners` on both DM flavours, plus `local_indices` and `global_indices`), and they already use exactly this vocabulary. 
Everything else returns a scalar, a range, or a PETSc object, so there is nothing further to unify here.

Note that `center` and `vertex` are themselves `NamedTuple`s keyed by axis, not `NTuple`s. Under dimension-correct returns a 2D DM yields `(x = …, y = …)` with no `z`:

```julia
local_indices(dm3d).center     # (x = 2:9, y = 2:9, z = 2:9)
local_indices(dm2d).center     # (x = 2:9, y = 2:9)
```

Results are **dimension-correct**: a 2D DM returns `CartesianIndex{2}` and 2-tuples.
This is a semantic break, covered in [§16](#16.-Breaking-changes-without-a-shim).

### 12.1 Index base

> Every index the high-level layer accepts or returns is 1-based. `LibPETSc` is 0-based.

There is no opt-out. `ownership_range` currently takes `base_one::Bool = true`, the one function in the package whose index convention is a runtime choice; the keyword is deprecated in v0.5 and removed in v0.6. 
Code wanting 0-based ranges should call `LibPETSc` directly, which is what that layer is for.

```julia
ownership_range(A)                    # 1-based, always
ownership_range(A; base_one = false)  # deprecated in 0.5, gone in 0.6
```

The invariant therefore holds unconditionally only from v0.6.

## 13. Exports

Only types and construction entry points are exported. Verbs and accessors stay qualified:

```julia
export DMDA, DMStag, DMPlex
export PetscVec, PetscMat, PetscOptions
export KSP, SNES, TS
export petsclibs
```

```julia
using PETSc
dm = DMStag(petsclib, comm, …)   # exported
c  = PETSc.corners(dm)           # qualified
PETSc.solve!(x, ksp, b)          # qualified
```

`initialize` and `finalize` are **not** exported: `Base.finalize` already exists, and shadowing it would be worse than typing `PETSc.finalize`.

Two reasons for staying narrow. `PETSc.solve!` has no `CommonSolve` dependency, so exporting it would make `using PETSc, LinearSolve` ambiguous for a common combination.
And adding an export later is non-breaking while removing one is not, so a narrow list keeps the option open.

### 13.1 Deferred: the `public` keyword

Not decided by this PR, and recorded here so the choice is made deliberately rather than by default.

Julia 1.11 added `public`, which marks a name as API without exporting it. 
That is exactly the distinction this package needs, because exporting only types ([§13](#13.-Exports)) leaves "unexported" unable to separate the API from internals. 
With `public`, the split would live in code:

```julia
export DMDA, DMStag, DMPlex, PetscVec, PetscMat, KSP, SNES, TS, PetscOptions
public corners, ghost_corners, local_indices, solve!, assemble!, destroy!, …
```

`names(PETSc)` would then report the API directly, `scripts/api_surface.jl` could check against declared `public` names instead of scanning this document, and the rename table would stop doubling as the register of what is public.

The cost is support. `Project.toml` pins `julia = "^1.10"`, and 1.10 is the current LTS, so adopting `public` means requiring 1.11 and dropping LTS users. 
That is a maintainer decision about the package's support policy, not a naming decision.

If it is ever taken, a breaking release is the cheapest moment: v0.5 already requires users to act. 
Until then the rename table is the register, and [§1.1](#1.1-What-these-rules-cover) describes how that works.

## 14. Errors

Argument problems raise standard Julia exceptions. `@assert` is reserved for invariants that cannot fail unless the package itself is wrong, and must never validate user input: it carries no useful message and is not guaranteed to run.

| Condition | Exception |
|---|---|
| Wrong argument value or type | `ArgumentError` |
| Mismatched sizes or shapes | `DimensionMismatch` |
| Library not initialized | `PetscNotInitialized` |
| Error returned by PETSc itself | `PetscError` (existing) |

```julia
length(A) == prod(sz) ||
    throw(DimensionMismatch("array has \$(length(A)) entries, expected \$(prod(sz))"))

T === petsclib.PetscScalar ||
    throw(ArgumentError("scalar type \$T does not match the library's \$(petsclib.PetscScalar)"))

isinitialized(lib) || throw(PetscNotInitialized(lib))
```

v0.4 has 43 `@assert` in the high-level layer. They resolve as:

| Kind | Count | Becomes |
|---|---|---|
| Size or length mismatch | 14 | `DimensionMismatch` |
| Library not initialized | 11 | `PetscNotInitialized` |
| Scalar or integer type mismatch | 8 | `ArgumentError` |
| DM flavour check (`gettype(dm) == "da"`) | 9 | deleted; dispatch enforces it ([§5.3](#5.3-DM-flavour-is-a-type,-not-a-string)) |

The last row is the point: nine of these checks exist only because one DM type had to police itself at runtime. 
Giving DM flavour a type deletes them rather than converting them.

## 15. Docstrings and discoverability

Renaming thin wrappers costs discoverability: a user who knows `PetscFECopyQuadrature` cannot grep for it once it is `copy_quadrature!`. 
Every high-level docstring must therefore carry an external link to the C function it wraps, using the existing `_doc_external` helper:

```julia
"""
    corners(dm::DMStag)

…

# External Links
$(_doc_external("DMSTAG/DMStagGetCorners"))
"""
```

CI fails on a high-level docstring with no `_doc_external` entry. 
The C-name-to-Julia-name index page is generated from those entries, so the lookup table maintains itself.

## 16. Breaking changes without a shim

`@deprecate` translates names, not semantics. These changes have no shim and must be
read before upgrading:

| Change | Symptom |
|---|---|
| Dimension-correct returns ([§12](#12.-Return-values)) | `corners(dm2d).size[3]` returned `1`, now throws `BoundsError`. `c.lower` is `CartesianIndex{2}`, not `{3}` |
| `DMStag`/`DMDA`/`DMPlex` return concrete types ([§5.3](#5.3-DM-flavour-is-a-type,-not-a-string)) | Code annotated `::PetscDM` no longer matches. Use `AbstractPetscDM` |
| `dm(ksp)` returns a Union ([§5.4](#5.4-DMs-of-unknown-provenance)) | Type-unstable at the boundary; add a function barrier in hot code |
| Type names are `Symbol` ([§3.1](#3.1-What-accessors-return-and-setters-take)) | `type(ksp) == "gmres"` is now false; compare against `:gmres`. Setters still accept a `String` only through the v0.5 shim |
| `@assert` replaced by typed exceptions ([§14](#14.-Errors)) | Code catching `AssertionError` must catch `ArgumentError`, `DimensionMismatch` or `PetscNotInitialized` |
| Arguments reordered ([§8](#8.-Argument-order)) | `dm_project_function!` and `dm_project_field!` took the written vector **last**; it moves to first. `dm_global_to_local!`/`dm_local_to_global!` took the DM last; it moves after the written vector. A shim can forward these, but any call written positionally against the old order and passed through `invoke` or a function reference will not be caught |

## 17. Migration

Shims live in `src/deprecations.jl`, separate from the existing `src/deprecated/` tree, which holds an older and unrelated API. 
They are removed in v0.6.

```julia
@deprecate getcorners_dmda(dm)        corners(dm)
@deprecate getcorners_dmstag(dm)      corners(dm)
@deprecate getghostcorners(dm)        ghost_corners(dm)
@deprecate getDM(ksp)                 dm(ksp)
@deprecate destroy(obj)               destroy!(obj)
```

Two shims are not plain renames. `ownership_range` keeps `base_one` in v0.5 and warns when it is passed, and `set_type!` accepts a `String` and warns, converting to `Symbol`.
Both are removed in v0.6.

The main test suite is converted to the new names, so CI exercises the API that ships.
`test/test_deprecations.jl` calls every shim under `@test_deprecated`, so the shims are covered too rather than rotting until they are deleted.

### Rename table

#### `dm.jl`

| v0.4 | v0.5 |
|---|---|
| `destroy` | `destroy!` |
| `getinfo` | `info` |
| `getcorners`, `getcorners_dmda` | `corners` |
| `getghostcorners`, `getghostcorners_dmda` | `ghost_corners` |
| `dm_local_to_global`, `dm_local_to_global!` | `local_to_global`, `local_to_global!` |
| `dm_global_to_local`, `dm_global_to_local!` | `global_to_local`, `global_to_local!` |
| `setuniformcoordinates_dmda!` | `set_uniform_coordinates!` |
| `coordinatesDMLocalVec` | `local_coordinates` |
| `getlocalcoordinatearray` | `local_coordinate_array` |
| `MatAIJ` | `PetscMat` constructor |
| `DMGlobalVec` | `global_vec` (merged, see below) |
| `DMLocalVec` | `local_vec` (merged, see below) |
| `getdimension` | `Base.ndims` |
| `setfromoptions!` | `set_from_options!` |

`DMGlobalVec` in `dm.jl` and `dm_create_global_vec` in `dmplex.jl` both call `DMCreateGlobalVector`; the local pair duplicates likewise. 
Each pair collapses to one name, so two shims point at each replacement.

#### `dmstag.jl`

| v0.4 | v0.5 |
|---|---|
| `getcorners_dmstag` | `corners` (method on `DMStag`) |
| `getghostcorners_dmstag` | `ghost_corners` (method on `DMStag`) |
| `local_indices_dmstag` | `local_indices` |
| `global_indices_dmstag` | `global_indices` |
| `setuniformcoordinates_stag!` | `set_uniform_coordinates!` |
| `DMStagDOF_Slot` | `dof_slot` |
| `to_petscint_tuple` | unchanged (internal) |

#### `dmda.jl`

| v0.4 | v0.5 |
|---|---|
| `reshapelocalarray` | `reshape_local_array` |
| `localinteriorlinearindex` | `local_interior_linear_index` |
| `dmda_star_fd_coloring` | `star_fd_coloring` |
| `ndofs` | unchanged (closed list) |

#### `dmplex.jl`

| v0.4 | v0.5 |
|---|---|
| `isplexsimplex` | `issimplex` |
| `plexdistribute!` | `distribute!` |
| `petsc_setname!` | `set_name!` |
| `getds` | `ds` |
| `createds!` | `create_ds!` |
| `getlabel` | `label` |
| `dm_project_function!` | `project_function!` |
| `dm_project_field!` | `project_field!` |
| `dm_compute_l2diff` | `l2diff` |
| `dm_create_global_vec` | `global_vec` |
| `dm_create_local_vec` | `local_vec` |
| `dm_set_auxiliary_vec!` | `set_auxiliary_vec!` |
| `dm_coarsen_hook_add!` | `add_coarsen_hook!` |
| `dm_copy_disc!` | `copy_disc!` |
| `dm_get_coarse` | `coarse_dm` |
| `fe_copy_quadrature!` | `copy_quadrature!` |
| `mat_null_space_create` | `mat_nullspace_create` (free function, prefix kept) |
| `mat_set_null_space!` | `set_nullspace!` |
| `mat_null_space_destroy!` | `destroy!` |
| `fe_create_default`, `fe_create_lagrange` | unchanged (free functions) |
| `vtk_save!`, `vtk_save_fields!` | `save_vtk!` (drops `petsclib`, see [§8](#8.-Argument-order)) |
| `vtk_merge_tensor!` | unchanged (free function, no PETSc object) |
| `setfield!` | `set_field!` (resolves the `Base.setfield!` shadow) |
| `dmclone` | `clone` |
| `plex_set_snes_local_fem!` | `set_snes_local_fem!` |
| `snes_set_jacobian_null_space!` | `set_jacobian_nullspace!` |
| `fe_compose_constant_null_space!` | `compose_constant_nullspace!` |
| `add_boundary!`, `add_natural_boundary!` | unchanged |
| `create_split_boundary_labels!` | unchanged |
| `set_constants!`, `set_exact_solution!` | unchanged |
| `set_residual!`, `set_jacobian!`, `set_jacobian_preconditioner!` | unchanged |
| `@petsc_residual_fn`, `@petsc_jacobian_fn` | `@residual_fn`, `@jacobian_fn` |
| `@petsc_bd_fn`, `@petsc_simple_fn` | `@bd_fn`, `@simple_fn` |

#### `ksp.jl`, `snes.jl`

| v0.4 | v0.5 |
|---|---|
| `PetscKSP`, `PetscSNES` (types) | `KSP`, `SNES` |
| `KSP(…)`, `SNES(…)` (factories) | type constructors |
| `getDM` | `dm` |
| `setDM!` | `set_dm!` |
| `get_solution` | `solution` |
| `gettype` | `type` |
| `destroy` | `destroy!` |
| `setcomputeoperators!` | `set_compute_operators!` |
| `setcomputerhs!` | `set_compute_rhs!` |
| `setfunction!` | `set_function!` |
| `setjacobian!` | `set_snes_jacobian!` (see note) |

`setjacobian!` and `dmplex.jl`'s `set_jacobian!` share an English word and nothing else:

```julia
setjacobian!(updateJ!, snes, J, PJ)                 # registers a callback
set_jacobian!(ds, fieldI, fieldJ, g0, g1, g2, g3)   # pointwise weak-form kernels
```

They stay separate names. 
Merging on a shared word is the error this document exists to prevent, and a merge here would put a callback and seven function pointers behind one generic. 
`set_snes_jacobian!` keeps a prefix under [§4](#4.-Object-prefixes) because without it the two collide; the callback argument leads it, per [§8.1](#8.1-Callbacks-come-first).

#### `vec.jl`, `mat.jl`

| v0.4 | v0.5 |
|---|---|
| `VecSeq`, `VecPtr` | `PetscVec` constructor |
| `MatCreateSeqAIJ`, `MatSeqAIJ`, `MatSeqDense`, `MatSeqAIJWithArrays` | `PetscMat` constructor |
| `unsafe_localarray`, `wrap_localarray` | `unsafe_local_array`, `wrap_local_array` |
| `acquire_petsc_local_array` | `acquire_local_array` |
| `release_petsc_local_array` | `release_local_array` |
| `get_petsc_arrays` | `local_arrays` |
| `restore_petsc_arrays` | `restore_local_arrays!` |
| `withlocalarray!` | `with_local_array!` |
| `ghostupdate!` | `ghost_update!` |
| `ghostupdatebegin!`, `ghostupdateend!` | `ghost_update_begin!`, `ghost_update_end!` |
| `ownershiprange` | `ownership_range` (drops `base_one`, see [§12.1](#12.1-Index-base)) |
| `setvalues!` | `set_values!` |
| `addindex!` | `add_index!` |
| `determine_memtype` | `memtype` |
| `as_petsc_vec` | `PetscVec` constructor |
| `array_type`, `memtype_backend` | unchanged |
| `make_local_array` | unchanged (internal) |
| `assemble!`, `setup!` | unchanged |
| `destroy` | `destroy!` |

#### `init.jl`

| v0.4 | v0.5 |
|---|---|
| `initialized` | `isinitialized` |
| `finalized` | `isfinalized` |
| `check_petsc_wrappers_version` | `check_wrappers_version` |
| `scalartype`, `inttype` | unchanged (closed list) |
| `initialize`, `finalize` | unchanged, not exported ([§13](#13.-Exports)) |
| `check_initialized` | unchanged (internal, see [§14](#14.-Errors)) |
| `isdestroyable` | unchanged (internal) |

#### Internals

Not part of the API, so no shims. 
They drop the `_` prefix, which was standing in for "internal" and is not how Julia expresses that:

| v0.4 | v0.5 |
|---|---|
| `_petsc_link` | `petsc_link` |
| `_petsc_subst` | `petsc_subst` |
| `_vtk_merge_one_tensor!` | `vtk_merge_one_tensor!` |
| `_build_petsc_options` | `build_petsc_options` |
| `_ensure_library_handle` | `ensure_library_handle` |
| `_ensure_mpi_initialized` | `ensure_mpi_initialized` |
| `_library_ptr` | `library_ptr` |
| `_post_initialize` | `post_initialize` |
| `_release_library_handle` | `release_library_handle` |
| `_doc_external`, `_lib_handles`, `_petsc_program_name` | prefix dropped likewise |

The exceptions keep their underscore, because there it separates an inner worker from the wrapper of the same name rather than marking visibility:

| v0.4 | v0.5 |
|---|---|
| `_mul!` | unchanged (worker for `mul!`) |
| `_unsafe_localarray` | `_unsafe_local_array` (worker for `unsafe_local_array`) |
| `get_petsc_arrays_impl` | `_local_arrays` (worker for `local_arrays`) |
| `restore_petsc_arrays_impl` | `_restore_local_arrays!` |

#### Types ([§5.5](#5.5-Abstract,-callback-and-wrapper-types))

| v0.4 | v0.5 |
|---|---|
| `AbstractPetscKSP`, `AbstractPetscSNES` | `AbstractKSP`, `AbstractSNES` |
| `AbstractPETScMemBackend` | `AbstractPetscMemBackend` |
| `Fn_KSPComputeRHS`, `Fn_KSPComputeOperators` | `KSPComputeRHSFn`, `KSPComputeOperatorsFn` |
| `Fn_SNESSetFunction`, `Fn_SNESSetJacobian` | `SNESSetFunctionFn`, `SNESSetJacobianFn` |
| `MatAT` | `MatOrTranspose` |
| `MatShell`, `MatOp`, `MatPtr`, `VecPtr` | unchanged |
| `AbstractPetscDS`, `PetscDS` | unchanged |
| `DMStagGetIndices` | removed (already deprecated in v0.4) |

#### `options.jl`, `sys.jl`

| v0.4 | v0.5 |
|---|---|
| `Options` (factory) | `PetscOptions` constructor |
| `getcomm` | `comm` |
| `typedget` | `parse_option` |
| `parse_options` | unchanged |

`typedget` operates on a plain `NamedTuple`, not on `PetscOptions`, and coerces the looked-up value to the type of the supplied default. 
`parse_option` names that, and pairs with `parse_options(args)` in the same file.

#### `init.jl`, `audit.jl`

| v0.4 | v0.5 |
|---|---|
| `set_petsclib` | `PetscLibType` constructor |
| `library_info` | returns a `NamedTuple`, printed via `show` |
| `audit_petsc_file` | `audit_file` |
| `set_library!`, `unset_library!` | unchanged ([§7](#7.-Mutation) global-state clause) |
| `audit_walk`, `audit_targets`, `audit_report`, `audit_creator`, `audit_destroyer`, `audit_callee`, `audit_argnames`, `audit_isbroadcast`, `audit_hasparseerror` | unchanged (internal) |

`library_info` printed a report and returned `nothing`, so its name promised data it never handed back. 
It now returns the values and gets a `show` method, which keeps the REPL output and makes the data reachable from tests:

```julia
info = library_info()
info.scalar          # Float64

julia> library_info()
Source  : LocalPreferences.toml
Path    : /usr/lib/libpetsc.so
```

### The leak auditor

`audit_file` (formerly `audit_petsc_file`) regex-matches the API's own names to pair object creations against `destroy` calls. 
The rename silently blinds it: after v0.5, `destroy` is `destroy!` and `MatSeqAIJ` is `PetscMat`, so the patterns stop matching and the auditor reports no leaks on leaking code. 
No test covers it today.

It is therefore rewritten in the same PR to match on the parsed AST (`Meta.parseall`, with line numbers from the emitted `LineNumberNode`s) against two name sets, one for creators and one for destroyers. 
The seven duplicated regex blocks collapse into a single walk, and four existing defects go with them:

- each regex block ends in `continue`, so only one event per line is recorded
- only full-line comments are skipped, so `# destroy!(A)` and `destroy` inside a string literal both count
- only bare and `PETSc.`-qualified `destroy` are recognised, so `LibPETSc.VecDestroy` reads as a leak
- the docstring hardcodes the names too, and drifts with the code

`test/test_audit.jl` audits a deliberately-leaking fixture and asserts the leak is found, so this class of drift fails CI on every future rename.

### Internal impact

The typed DM hierarchy requires widening autowrapped signatures from the concrete `PetscDM{PetscLib}` to `AbstractPetscDM{PetscLib}`: about 4000 occurrences, roughly 3500 of them in `src/autowrapped/DM_wrappers.jl`. 
This is a mechanical substitution over generated files. 
The generator in `wrapping/` must be updated to emit `AbstractPetscDM` so the change survives a regeneration.
