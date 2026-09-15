using Test
using PETSc, MPI
using PETSc: LibPETSc

if !Sys.iswindows()
    MPI.Initialized() || MPI.Init()
end

# ============================================================================
#   Typed DM hierarchy
# ============================================================================
#
# Flavour and dimension live in the type, so the corner and index queries
# dispatch instead of asking DMGetType and branching on the answer.

@testset "DM flavour is a type" begin
    dm_comm = LibPETSc.PETSC_COMM_SELF
    for petsclib in PETSc.petsclibs
        PETSc.initialize(petsclib)
        PetscInt = petsclib.PetscInt

        da = PETSc.DMDA(
            petsclib, dm_comm,
            (LibPETSc.DM_BOUNDARY_NONE, LibPETSc.DM_BOUNDARY_NONE),
            (PetscInt(6), PetscInt(4)),
            PetscInt(1), PetscInt(1), LibPETSc.DMDA_STENCIL_STAR,
        )
        stag = PETSc.DMStag(
            petsclib, dm_comm,
            (LibPETSc.DM_BOUNDARY_NONE,),
            (PetscInt(5),),
            (PetscInt(1), PetscInt(1)),
            PetscInt(1),
        )

        @test da isa PETSc.DMDA{<:Any, 2}
        @test stag isa PETSc.DMStag{<:Any, 1}
        @test da isa LibPETSc.AbstractPetscDM
        @test stag isa LibPETSc.AbstractPetscDM

        # Each flavour has one concrete return type, where getcorners used to
        # return a union shaped by a runtime string comparison. The size tuple
        # follows the library's PetscInt.
        @test Base.return_types(PETSc.getcorners, (typeof(da),))[1] ===
              NamedTuple{
                  (:lower, :upper, :size),
                  Tuple{
                      CartesianIndex{3},
                      CartesianIndex{3},
                      Tuple{PetscInt, PetscInt, PetscInt},
                  },
              }
        @test isconcretetype(Base.return_types(PETSc.getcorners, (typeof(stag),))[1])

        @test PETSc.getcorners(da).size == (6, 4, 1)
        @test size(da) == (6, 4, 1)
        @test haskey(PETSc.getcorners(stag), :nextra)

        # narrow resolves a handle the low-level creators hand back untyped.
        bare = LibPETSc.PetscDM{typeof(petsclib)}(da.ptr, da.age)
        @test PETSc.narrow(bare) isa PETSc.DMDA{<:Any, 2}
        @test PETSc.narrow(da) === da

        # An untyped handle still reaches the flavour API, by resolving itself.
        @test PETSc.getcorners(bare) == PETSc.getcorners(da)

        # DMDAGetInfo and DMDAGetDof do not check the flavour they are handed:
        # on a DMStag they report success and return whatever was in the output
        # buffers. Dispatch is what stops the call.
        @test_throws MethodError PETSc.getinfo(stag)
        @test_throws MethodError PETSc.ndofs(stag)
        @test PETSc.getinfo(da).dim == 2

        PETSc.destroy(da)
        PETSc.destroy(stag)
        PETSc.finalize(petsclib)
    end
end

@testset "DM accessors hand back borrowed handles" begin
    dm_comm = LibPETSc.PETSC_COMM_SELF
    for petsclib in PETSc.petsclibs
        PETSc.initialize(petsclib)
        PetscInt = petsclib.PetscInt

        da = PETSc.DMDA(
            petsclib, dm_comm,
            (LibPETSc.DM_BOUNDARY_NONE,), (PetscInt(8),),
            PetscInt(1), PetscInt(1),
        )
        ksp = PETSc.KSP(da)

        as_object(dm) = Base.unsafe_convert(LibPETSc.PetscObject, Ptr{Cvoid}(dm.ptr))
        refs() = LibPETSc.PetscObjectGetReference(petsclib, as_object(da))

        borrowed = PETSc.getDM(ksp)
        @test borrowed isa PETSc.DMDA
        @test !PETSc.owns(borrowed)
        @test PETSc.owns(da)

        # KSPGetDM takes no reference, so destroying its result would release
        # one the KSP still holds. destroy must decline instead.
        before = refs()
        PETSc.destroy(borrowed)
        @test refs() == before
        @test borrowed.ptr != C_NULL
        @test PETSc.getcorners(da).size == (8, 1, 1)

        PETSc.destroy(da)
        PETSc.finalize(petsclib)
    end
end
