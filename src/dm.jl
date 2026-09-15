import .LibPETSc: AbstractPetscDM, PetscDM, CDM

# ============================================================================
#   DM flavours
# ============================================================================
#
# PETSc resolves a DM's flavour at runtime through DMGetType. Carrying it in the
# Julia type instead lets ordinary dispatch pick the method, so the corner and
# index queries stop branching on a string and each one has a single return
# type.
#
# Dimension is a parameter only where a method dispatches on it. That holds for
# DMDA and DMStag, whose creation and corner paths are written per dimension. It
# does not hold for DMPlex: nothing dispatches on a plex's dimension, and
# `DMPlex(petsclib, comm)` leaves it unset until the mesh is built, so a plex
# reports it through `getdimension` instead.
#
# `own` records whether this wrapper is responsible for destroying the handle.
# A DM reached from another object is borrowed: the accessor takes no reference,
# so destroying it releases one the owner still holds, and enough of those free
# the DM while the owner is still pointing at it.

mutable struct DMDA{PetscLib, N} <: AbstractPetscDM{PetscLib}
    ptr::CDM
    age::Int
    own::Bool
end

mutable struct DMStag{PetscLib, N} <: AbstractPetscDM{PetscLib}
    ptr::CDM
    age::Int
    own::Bool
end

mutable struct DMPlex{PetscLib} <: AbstractPetscDM{PetscLib}
    ptr::CDM
    age::Int
    own::Bool
end

owns(dm::Union{DMDA, DMStag, DMPlex}) = dm.own

"""
    narrow(dm::PetscDM; own = false)

Return `dm` as the concrete type matching its PETSc flavour, so later calls
dispatch rather than query.

`LibPETSc.DMCreate` and the accessors that hand back a DM belonging to another
object return the untyped `PetscDM`, and this is what resolves it. The result is
a second handle onto the same PETSc object, borrowed unless `own` says
otherwise, so destroying either invalidates the other.

The return type is a `Union` wider than inference will split, which costs one
dynamic dispatch. Narrow once outside a hot loop and pass the result through a
function barrier.

# External Links
$(_doc_external("DM/DMGetType"))
$(_doc_external("DM/DMGetDimension"))
"""
function narrow(dm::PetscDM{PetscLib}; own::Bool = false) where {PetscLib}
    dm.ptr == C_NULL && return dm
    # A DM straight from DMCreate has no type yet, and DMGetType raises rather
    # than answering. It has no flavour to narrow to, so hand it back as it is.
    type = try
        LibPETSc.DMGetType(PetscLib, dm)
    catch
        return dm
    end
    if type == "da"
        N = Int(LibPETSc.DMGetDimension(PetscLib, dm))
        return DMDA{PetscLib, N}(dm.ptr, dm.age, own)
    elseif type == "stag"
        N = Int(LibPETSc.DMGetDimension(PetscLib, dm))
        return DMStag{PetscLib, N}(dm.ptr, dm.age, own)
    elseif type == "plex"
        return DMPlex{PetscLib}(dm.ptr, dm.age, own)
    end
    # A flavour with no type of its own comes back unchanged.
    return dm
end

# Already narrowed.
narrow(dm::AbstractPetscDM; own::Bool = false) = dm

# A handle built through the low-level creators arrives untyped, and calling the
# flavour-specific API on it stays legal. Resolving it here costs a DMGetType
# query per call, which the typed constructors and a single `narrow` both avoid.
function _flavoured(dm::PetscDM{PetscLib}, what::AbstractString) where {PetscLib}
    narrowed = narrow(dm)
    narrowed isa PetscDM && throw(
        ArgumentError(
            "$what needs a DM whose flavour has a Julia type (DMDA, DMStag or " *
            "DMPlex); this handle has none",
        ),
    )
    return narrowed
end

# Custom display for REPL
function Base.show(io::IO, v::AbstractPetscDM{PetscLib}) where {PetscLib}
    if v.ptr == C_NULL
        print(io, "PETSc DM (null pointer)")
        return
    end
    # DMGetType internally calls DMInitializePackage which queries the PETSc
    # options database.  Calling it before PETSc is initialised causes a C-level
    # SIGSEGV that cannot be caught with try/catch.
    if !initialized(PetscLib)
        print(io, "PETSc DM (PETSc not initialized)")
        return
    end
    try
        ty = LibPETSc.DMGetType(PetscLib, v)
        di = LibPETSc.DMGetDimension(PetscLib, v)
        print(io, "PETSc DM $ty object in $di dimensions")
    catch
        print(io, "PETSc DM (type not set)")
    end
    return nothing
end


"""
    destroy(dm::AbstractPetscDM)

Destroy a DM object and release associated resources.

This function is typically called automatically via finalizers when the object
is garbage collected, but can be called explicitly to free resources immediately.

Does nothing on a borrowed handle, one reached from another object or produced
by [`narrow`](@ref). Such an accessor takes no reference, so destroying its
result releases one the owner still holds.

# External Links
$(_doc_external("DM/DMDestroy"))
"""
function destroy(dm::AbstractPetscDM{PetscLib}) where {PetscLib}
    owns(dm) || return nothing
    if isdestroyable(dm, PetscLib)
        LibPETSc.DMDestroy(PetscLib, dm)
    end
    dm.ptr = C_NULL
    return nothing
end



"""
    getinfo(dm::DMDA)

Get information about a DMDA.

Restricted to a `DMDA` because `DMDAGetInfo` does not check the flavour it is
given: on a `DMStag` it fills the output buffers with whatever was in them and
reports success, so the caller gets uninitialised values rather than an error.

# Returns

A `NamedTuple` with the following fields:
- `dim`: Dimension of the `DMDA` (1, 2, or 3)
- `global_size`: Tuple with global dimensions in each direction of the array
- `mpi_proc_size`: Tuple with number of MPI processes in each direction
- `dof`: Degrees of freedom per node
- `stencil_width`: Width of the stencil
- `boundary_type`: Tuple with boundary types in each direction
- `stencil_type`: Stencil type, either `DMDA_STENCIL_STAR` or `DMDA_STENCIL_BOX`

# External Links
$(_doc_external("DMDA/DMDAGetInfo"))
"""
function getinfo(dm::DMDA{PetscLib}) where {PetscLib}

    dim, M, N, P, m, n, p, dof, s, bx, by, bz, st = LibPETSc.DMDAGetInfo(PetscLib, dm)
    global_size   = (M,N,P)
    mpi_proc_size = (m,n,p)
    boundary_type = (bx,by,bz)
    stencil_width = s
    stencil_type  = st
               

	return (;dim,global_size,mpi_proc_size,dof,s,boundary_type,stencil_width,stencil_type)
end

getinfo(dm::PetscDM) = getinfo(_flavoured(dm, "getinfo"))

# Shape the DMDA corner triples the compiler can see through. Building a Vector
# and splatting it hides the length, so `lower` and `upper` used to infer as
# `Any` and every call allocated.
function _corner_tuple(xs, ys, zs, xm, ym, zm, PetscInt)
    lower = CartesianIndex(
        Int(xs) + 1,
        Int(ys) + 1,
        Int(zs) + 1,
    )
    local_size = (PetscInt(xm), PetscInt(ym), PetscInt(zm))
    upper = CartesianIndex(
        Int(xs) + Int(xm),
        Int(ys) + Int(ym),
        Int(zs) + Int(zm),
    )
    return (lower = lower, upper = upper, size = local_size)
end

"""
    lower, upper, size = getcorners(da::DMDA)

Returns a `NamedTuple` with the global indices (excluding ghost points) of the
`lower` and `upper` corners as well as the `size`.

# External Links
$(_doc_external("DMDA/DMDAGetCorners"))
"""
function getcorners(dm::DMDA{PetscLib}) where {PetscLib}
    PetscInt = inttype(PetscLib)
    xs, ys, zs, xm, ym, zm = LibPETSc.DMDAGetCorners(PetscLib, dm)
    return _corner_tuple(xs, ys, zs, xm, ym, zm, PetscInt)
end

"""
    lower, upper, size = getghostcorners(da::DMDA)

Returns a `NamedTuple` with the global indices (including ghost points) of the
`lower` and `upper` corners as well as the `size` of the local part of the domain.

# External Links
$(_doc_external("DMDA/DMDAGetGhostCorners"))
"""
function getghostcorners(dm::DMDA{PetscLib}) where {PetscLib}
    PetscInt = inttype(PetscLib)
    xs, ys, zs, xm, ym, zm = LibPETSc.DMDAGetGhostCorners(PetscLib, dm)
    return _corner_tuple(xs, ys, zs, xm, ym, zm, PetscInt)
end

# The suffixed spellings predate flavour dispatch and stay as forwarders, since
# the examples still call them.
getcorners_dmda(dm::DMDA) = getcorners(dm)
getghostcorners_dmda(dm::DMDA) = getghostcorners(dm)

getcorners(dm::PetscDM) = getcorners(_flavoured(dm, "getcorners"))
getghostcorners(dm::PetscDM) = getghostcorners(_flavoured(dm, "getghostcorners"))


"""
    setup!(dm::DM)

# External Links
$(_doc_external("DM/DMSetUp"))
"""
setup!(dm::AbstractPetscDM{PetscLib}) where {PetscLib} = LibPETSc.DMSetUp(PetscLib, dm)


"""
    setfromoptions!(dm::AbstractPetscDM)

Sets the global options to the `dm`    
# External Links
$(_doc_external("DM/DMSetFromOptions"))
"""
setfromoptions!(dm::AbstractPetscDM{PetscLib}) where {PetscLib} = LibPETSc.DMSetFromOptions(PetscLib, dm)



"""
    empty(da::AbstractPetscDM)

return an uninitialized `DMDA` struct.
"""
Base.empty(da::AbstractPetscDM{PetscLib}) where {PetscLib} = PetscDM{PetscLib}(C_NULL, da.age)


"""
    v::PetscVec = DMLocalVec(dm::AbstractPetscDM{PetscLib}) where {PetscLib}

Returns a local vector `v` from the `dm` object.
"""
DMLocalVec(dm::AbstractPetscDM{PetscLib}) where {PetscLib} = LibPETSc.DMCreateLocalVector(getlib(PetscLib), dm)

"""
    v::PetscVec = DMGlobalVec(dm::AbstractPetscDM{PetscLib}) where {PetscLib}

Returns a global vector `v` from the `dm` object.
"""
DMGlobalVec(dm::AbstractPetscDM{PetscLib}) where {PetscLib} = LibPETSc.DMCreateGlobalVector(getlib(PetscLib), dm)

"""
    dm_local_to_global!(local_vec, global_vec, dm, mode = INSERT_VALUES)

Transfer values from the `local_vec` to the `global_vec` associated with the `dm` object.

# Arguments
- `local_vec::AbstractPetscVec`: Local vector (source)
- `global_vec::AbstractPetscVec`: Global vector (destination)
- `dm::AbstractPetscDM`: DM object
- `mode::InsertMode`: Insert mode, either `INSERT_VALUES` or `ADD_VALUES`

# External Links
$(_doc_external("DM/DMLocalToGlobal"))

"""
function dm_local_to_global!(   local_vec::AbstractPetscVec{PetscLib},
                                global_vec::AbstractPetscVec{PetscLib},
                                   dm::AbstractPetscDM{PetscLib},
                                   mode::InsertMode = INSERT_VALUES) where {PetscLib}

    LibPETSc.DMLocalToGlobalBegin(PetscLib, dm, local_vec, mode, global_vec)
    LibPETSc.DMLocalToGlobalEnd(PetscLib, dm, local_vec,  mode, global_vec)
    return nothing
end


"""
    dm_global_to_local!(global_vec, local_vec, dm, mode = INSERT_VALUES)

Transfer values from the `global_vec` to the `local_vec` associated with the `dm` object,
including ghost point values from neighboring processes.

# Arguments
- `global_vec::AbstractPetscVec`: Global vector (source)
- `local_vec::AbstractPetscVec`: Local vector (destination)
- `dm::AbstractPetscDM`: DM object
- `mode::InsertMode`: Insert mode, either `INSERT_VALUES` or `ADD_VALUES`

# External Links
$(_doc_external("DM/DMLocalToGlobal"))
"""
function dm_global_to_local!(global_vec::AbstractPetscVec{PetscLib},
                              local_vec::AbstractPetscVec{PetscLib},
                                     dm::AbstractPetscDM{PetscLib},
                                   mode::InsertMode = INSERT_VALUES) where {PetscLib}

    LibPETSc.DMGlobalToLocalBegin(getlib(PetscLib), dm, global_vec, mode, local_vec)
    LibPETSc.DMGlobalToLocalEnd(getlib(PetscLib), dm, global_vec,  mode, local_vec)
    return nothing
end


"""
    setuniformcoordinates!(
        da::DMDA
        xyzmin::NTuple{N, Real},
        xyzmax::NTuple{N, Real},
    ) where {N}

Set uniform coordinates for the `da` using the lower and upper corners defined
by the `NTuple`s `xyzmin` and `xyzmax`. If `N` is less than the dimension of the
`da` then the value of the trailing coordinates is set to `0`.

# External Links
$(_doc_external("DMDA/DMDASetUniformCoordinates"))
"""
function setuniformcoordinates_dmda!(
    da::DMDA{PetscLib},
    xyzmin::NTuple{N, Real},
    xyzmax::NTuple{N, Real},
) where {N, PetscLib}
    PetscReal = PetscLib.PetscReal
    xmin = PetscReal(xyzmin[1])
    xmax = PetscReal(xyzmax[1])

    ymin = (N > 1) ? PetscReal(xyzmin[2]) : PetscReal(0)
    ymax = (N > 1) ? PetscReal(xyzmax[2]) : PetscReal(0)

    zmin = (N > 2) ? PetscReal(xyzmin[3]) : PetscReal(0)
    zmax = (N > 2) ? PetscReal(xyzmax[3]) : PetscReal(0)


    LibPETSc.DMDASetUniformCoordinates(
        PetscLib,
        da,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
    )
    return da
end

setuniformcoordinates_dmda!(da::PetscDM, xyzmin::NTuple, xyzmax::NTuple) =
    setuniformcoordinates_dmda!(
        _flavoured(da, "setuniformcoordinates_dmda!"), xyzmin, xyzmax,
    )

"""
    coordinatesDMLocalVec(dm::AbstractDM)

Gets a local vector with the coordinates associated with `dm`.

Note that the returned vector is borrowed from the `dm` and is not a new vector.

# External Links
$(_doc_external("DM/DMGetCoordinatesLocal"))
"""
function coordinatesDMLocalVec(dm::AbstractPetscDM{PetscLib}) where {PetscLib}
    petsclib = getlib(PetscLib)
    coord_vec = DMLocalVec(dm)
    LibPETSc.DMGetCoordinatesLocal(PetscLib, dm, coord_vec)

    return coord_vec
end

"""
    getlocalcoordinatearray(da::AbstractPetscDM)

Return coordinate arrays for the local portion of the domain.

The returned arrays are `OffsetArray`s that can be addressed using global indices,
accounting for ghost points.

# External Links
$(_doc_external("DM/DMGetCoordinatesLocal"))
"""
function getlocalcoordinatearray(da::AbstractPetscDM{PetscLib}) where {PetscLib}
    # retrieve local coordinates
    coord_vec = coordinatesDMLocalVec(da)
    # array
    array1D = unsafe_localarray(coord_vec; read = true, write = false)
    dim = [PetscLib.PetscInt(0)]
    dim = LibPETSc.DMGetCoordinateDim(PetscLib, da)
    dim = dim[1]
    corners = getghostcorners(da)

    return reshapelocalarray(array1D, da, dim)
end


gettype(dm::AbstractPetscDM{PetscLib}) where {PetscLib} = LibPETSc.DMGetType(PetscLib,dm)

"""
    getdimension(dm::AbstractPetscDM)

Return the topological dimension of the `dm`

# External Links
$(_doc_external("DM/DMGetDimension"))
"""
getdimension(dm::AbstractPetscDM{PetscLib}) where PetscLib = LibPETSc.DMGetDimension(PetscLib,dm)


"""
    size(dm::DMDA)
    size(dm::DMStag)

Return the global size of a DM object as a tuple `(M, N, P)`, where unused
dimensions are 1.

# External Links
$(_doc_external("DMDA/DMDAGetInfo"))
$(_doc_external("DMStag/DMStagGetGlobalSizes"))
"""
function Base.size(dm::DMDA{PetscLib}) where {PetscLib}
    _, M, N, P, _ = LibPETSc.DMDAGetInfo(PetscLib, dm)
    return (M, N, P)
end

Base.size(dm::DMStag{PetscLib}) where {PetscLib} =
    LibPETSc.DMStagGetGlobalSizes(PetscLib, dm)

Base.size(dm::PetscDM) = size(_flavoured(dm, "size"))

#=
"""
    dm_local_to_global(dm, x_L, x_G, mode = INSERT_VALUES)

Transfer values from the local vector `x_L` to the global vector `x_G`.

# Arguments
- `dm`: The DM object
- `x_L`: Local vector (source)
- `x_G`: Global vector (destination)
- `mode`: `INSERT_VALUES` (default) or `ADD_VALUES`

# External Links
$(_doc_external("DM/DMLocalToGlobal"))
"""
function dm_local_to_global(dm::PetscDM{PetscLib},
                             x_L::AbstractPetscVec{PetscLib},
                             x_G::AbstractPetscVec{PetscLib}, 
                             mode=LibPETSc.INSERT_VALUES) where {PetscLib}
    
    petsclib = getlib(PetscLib)
    LibPETSc.DMLocalToGlobalBegin(petsclib, dm, x_L, mode, x_G)
    LibPETSc.DMLocalToGlobalEnd(petsclib, dm, x_L, mode, x_G)
    
    return nothing
end
=#
#=
"""
    dm_global_to_local(dm, x_G, x_L, mode = INSERT_VALUES)

Transfer values from the global vector `x_G` to the local vector `x_L`,
including ghost point values from neighboring processes.

# Arguments
- `dm`: The DM object
- `x_G`: Global vector (source)
- `x_L`: Local vector (destination)
- `mode`: `INSERT_VALUES` (default) or `ADD_VALUES`

# External Links
$(_doc_external("DM/DMGlobalToLocal"))
"""
function dm_global_to_local(dm::PetscDM{PetscLib},
                             x_G::AbstractPetscVec{PetscLib},
                             x_L::AbstractPetscVec{PetscLib}, 
                             mode=LibPETSc.INSERT_VALUES) where {PetscLib}
    
    petsclib = getlib(PetscLib)
    LibPETSc.DMGlobalToLocalBegin(petsclib, dm, x_G, mode, x_L)
    LibPETSc.DMGlobalToLocalEnd(petsclib, dm, x_G, mode, x_L)

    return nothing
end
=#

"""
    MatAIJ(da::AbstractPetscDM)

Create a sparse matrix (AIJ format) with sparsity pattern determined by the DM.

# Returns

A `PetscMat` object compatible with vectors from the DM.

# External Links
$(_doc_external("DM/DMCreateMatrix"))
"""
function MatAIJ(da::AbstractPetscDM{PetscLib}) where {PetscLib}
    J = LibPETSc.DMCreateMatrix(getlib(PetscLib), da)
    return J
end

