#!/bin/bash
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup,modules
#SBATCH --output=%x_%j.out
# Run several variants of one figure point in one allocation, one after another.
# Usage: sbatch --account=A --job-name=NAME [size opts] run_point.sh pN variant...   (variant: jll, loc or nat)
# Submit from this directory; the options come from ../opts/pN.opts and ../opts/common.opts.
# JLL_ENV and LOCAL_ENV are the Julia environments of the two PETSc.jl variants, NATIVE the
# C program built with ../makefile.
# A variant may be listed several times for repetitions on the same nodes.
# Each run writes <variant>_<pN>_<jobid>.out, and repetitions <variant>_<pN>_<jobid>.<k>.out.
# The job stops when a run's iterations or L2 error differ from the first one.
set -euo pipefail
p=$1; shift
opts=$(cat ../opts/$p.opts ../opts/common.opts)
ex45=${PETSC_JL:-$HOME/PETSc.jl}/examples/dmda/ex45.jl
ref=""
declare -A count
for v in "$@"; do
    count[$v]=$(( ${count[$v]:-0} + 1 ))
    k=${count[$v]}
    case $v in
        jll) env=${JLL_ENV:-$HOME/petsc_bench_mpt}; prog=$ex45 ;;
        loc) env=${LOCAL_ENV:-$HOME/petsc_bench_local}; prog=$ex45 ;;
        nat) env=${JLL_ENV:-$HOME/petsc_bench_mpt}; prog=${NATIVE:-$HOME/petsc_bench/native/ex45_julia} ;;
        *) echo "unknown variant $v"; exit 1 ;;
    esac
    if (( k == 1 )); then log=${v}_${p}_${SLURM_JOB_ID}.out; else log=${v}_${p}_${SLURM_JOB_ID}.$k.out; fi
    echo "$v (run $k): started $(date)"
    BENCH_ENV=$env bash run.sh $prog $opts > $log 2>&1 || { echo "$v (run $k): failed or timed out, see $log, stopping"; exit 1; }
    # PETSc lists options nobody read; a misspelled option silently changes the solver.
    # ex45.jl reads the grid size (-Nx -Ny -Nz) itself, so PETSc lists those for Julia runs.
    unused=$(grep "^Option left:" $log | grep -vE "name:-N[xyz] " || true)
    if [[ -n $unused ]]; then
        echo "$unused"
        echo "UNUSED OPTION in $v (run $k), stopping"
        exit 1
    fi
    result=$(grep -E "iterations [0-9]+$|^L2 error" $log | sort -u)
    echo "$v: finished $(date)"
    echo "$result"
    # VecAYPX does no communication, so its max/min time ratio over the ranks is ~1.2 on
    # quiet nodes; a much larger ratio means some cores ran slow and the run is disturbed
    ratio=$(sed -n '/Event Stage 1: MG Apply/,/^---/p' $log | awk '$1 == "VecAYPX" {print $5}')
    echo "$v: VecAYPX max/min ratio $ratio"
    awk -v r="$ratio" 'BEGIN { exit !(r > 1.6) }' && echo "WARNING: $v looks disturbed (ratio $ratio > 1.6), rerun it"
    if [[ -z $ref ]]; then
        ref=$result
    elif [[ $result != "$ref" ]]; then
        echo "MISMATCH: $v differs from the first variant, stopping"
        exit 1
    fi
done
