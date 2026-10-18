#!/bin/bash
#! Slurm array for the full-grid prior-sensitivity map (cosmocompare).
#! One array task per (model, dataset) pair enumerated by grid_prior_sweep.py
#! (662 pairs as of this writing: indices 0-661). Each task reweights the chains
#! in place (no likelihood calls) and writes its own catalogue shard, so the array
#! is parallel-safe; merge the shards once at the end.
#!
#! dp465 (Seedcorn, "Mapping the Prior-Sensitivity ...") is on Peta4-Skylake,
#! CSD3's legacy Skylake system, reached via login.hpc.cam.ac.uk (not login-cpu,
#! which is the icelake system). The account/partition below follow the CSD3 house
#! pattern (DIRAC-DP192-CPU -> DIRAC-DP465-CPU; partition skylake); VERIFY on a
#! Peta4-Skylake login with `sacctmgr -n show assoc user=$USER | grep -i 465`
#! before submitting, and adjust if the exact strings differ. A fallback with
#! plenty of balance is --account=DIRAC-DP192-CPU --partition=icelake.
#!
#! Submit:  sbatch --array=0-661%64 grid_prior_sweep.slurm
#! Merge:   python grid_prior_sweep.py --outdir $HOME/cc_grid --merge

#SBATCH --job-name=cc_grid
#SBATCH --account=DIRAC-DP465-CPU
#SBATCH --partition=skylake
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

python $HOME/cosmocompare_src/scripts/grid_prior_sweep.py \
    --outdir $HOME/cc_grid \
    --task-index ${SLURM_ARRAY_TASK_ID} \
    --nsamples 1000 --n-bootstrap 1000 --n-mc 1000000 --seed 0
