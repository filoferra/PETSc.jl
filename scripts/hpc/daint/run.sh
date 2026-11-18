#!/bin/bash
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup,modules
#SBATCH --output=%x_%j.out
# Run a Julia script (*.jl) or a program on all ranks of the allocation.
# Usage: sbatch --account=A --job-name=NAME --partition=P --nodes=N --ntasks-per-node=T --time=HH:MM:SS run.sh program [args...]
# BENCH_ENV selects the Julia environment (default: the MPItrampoline one), PETSC_JL the
# PETSc.jl checkout whose commit is logged.
set -euo pipefail
BENCH_ENV=${BENCH_ENV:-$HOME/petsc_bench_mpt}
# MPItrampoline forwards MPI calls to this wrapper, built against the uenv's Cray MPICH
export MPITRAMPOLINE_LIB=${MPITRAMPOLINE_LIB:-$HOME/mpiwrapper/lib64/libmpiwrapper.so}
PETSC_JL=${PETSC_JL:-$HOME/PETSc.jl}
# One thread per rank: OpenMP in SuperLU_dist otherwise starts all-core thread teams in every rank
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# CPU runs need no GPU-aware MPI; the uenv turns it on, which initializes CUDA on first use. GPU runs: GPU_AWARE_MPI=1
export MPICH_GPU_SUPPORT_ENABLED=${GPU_AWARE_MPI:-0}
if [[ $1 == *.jl ]]; then
    # BENCH_ENV must be precompiled on the login node first: otherwise every rank compiles at once and the job hangs
    launcher=(julia --project=$BENCH_ENV)
    # Warm-up: load the packages once per node, so the ranks read the depot from the page cache,
    # as native PETSc reads its prebuilt binary. Not part of the measured start-up.
    echo "warm-up started $(date)"
    srun --nodes=$SLURM_NNODES --ntasks-per-node=1 --cpu-bind=none julia --project=$BENCH_ENV -e "using PETSc"
    echo "warm-up finished $(date)"
else
    launcher=()
fi
echo "job $SLURM_JOB_ID: $SLURM_NNODES nodes, $SLURM_NTASKS tasks, started $(date)"
echo "command: ${launcher[*]} $*"
echo "PETSc.jl commit: $(git -C $PETSC_JL --no-pager log -1 --oneline)"
echo "PETSc.jl local changes: $(git -C $PETSC_JL status --short | tr "\n" " ")"
# RUN_TIMEOUT [s] stops a hung run before it uses up the job's time limit on every node
echo "run timeout: ${RUN_TIMEOUT:-300} s"
status=0
timeout ${RUN_TIMEOUT:-300} srun --mpi=cray_shasta --cpu-bind=verbose,cores --distribution=block:cyclic "${launcher[@]}" "$@" || status=$?
if (( status == 124 )); then
    echo "TIMEOUT: run killed after ${RUN_TIMEOUT:-300} s $(date)"
    exit 1
elif (( status != 0 )); then
    echo "FAILED: srun exited with $status $(date)"
    exit $status
fi
echo "finished $(date)"
