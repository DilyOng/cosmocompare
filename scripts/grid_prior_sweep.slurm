#!/bin/bash
#! Slurm array for the full-grid prior-sensitivity map (cosmocompare).
#! One array task per (model, dataset) pair enumerated by grid_prior_sweep.py.
#! Each task reweights the chains in place (no likelihood calls) and writes its own
#! catalogue shard, so the array is parallel-safe; merge the shards once at the end.
#!
#! This runs on DIRAC-DP192-CPU / icelake (login-cpu.hpc.cam.ac.uk), where the full
#! 6-model grid (job cc_grid_38766236) was produced. DP465 is a *separate* cluster
#! (Peta4-Skylake, via login.hpc.cam.ac.uk): its "skylake" partition is not valid
#! from the icelake login, so it cannot be targeted here.
#!
#! Restrict the models with the MODELS env var (space-separated); unset = all.
#! The number of array tasks must equal the pair count printed by:
#!   grid_prior_sweep.py [--models $MODELS] --list
#!
#! Full grid:  sbatch --array=0-661%64 grid_prior_sweep.slurm
#! Neutrinos:  sbatch --array=0-142%64 --export=ALL,MODELS="mlcdm nlcdm Nlcdm" \
#!                 grid_prior_sweep.slurm        # 143 post-fix cells
#! Merge:      python grid_prior_sweep.py --outdir $HOME/cc_grid --merge

#SBATCH --job-name=cc_grid
#SBATCH --account=DIRAC-DP192-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -e
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
source /home/dlo26/rds/hpc-work/venv/tension_venv/bin/activate
export PYTHONPATH=$HOME/cosmocompare_src
export MPLBACKEND=Agg

#! OUTDIR defaults to the main catalogue; set it to a separate dir to avoid
#! overwriting existing results. NFRAC sets a dense width grid when given.
#! CENTRESCAN=1 runs the centre x width scan (one-parameter models only); NCENTRES
#! sets the number of centres.
python $HOME/cosmocompare_src/scripts/grid_prior_sweep.py \
    --outdir ${OUTDIR:-$HOME/cc_grid} \
    ${MODELS:+--models $MODELS} \
    ${NFRAC:+--n-fractions $NFRAC} \
    ${CENTRESCAN:+--centre-scan} \
    ${NCENTRES:+--n-centres $NCENTRES} \
    --task-index ${SLURM_ARRAY_TASK_ID} \
    --nsamples 1000 --n-bootstrap 1000 --n-mc 1000000 --seed 0
