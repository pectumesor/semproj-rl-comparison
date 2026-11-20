#!/usr/bin/env bash
set -euo pipefail

ETH_USERNAME=eazevedo
PROJECT_NAME=semproj
SCRATCH=/cluster/scratch/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=semproj
CONDA_ROOT=${HOME}/miniforge3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Report material in one W&B group (scripts/report_to_wandb.py): one SLURM job per section of the supervisor report,
# each one W&B run with the report's figures, room pictures, learning curves, tables and rollout videos of whole
# episodes from the saved checkpoints. Only reads the sweeps' seed folders; the training runs keep their groups.
# The figures are the ones embedded in the report PDF (pdfimages -png), copied to ${FIGURES} beforehand.
#     bash scripts/report_to_wandb.sh [group] [section ...]      (default: report_nov19, every section)
GROUP=${1:-report_nov19}
shift || true
SECTIONS=("$@")
if [ ${#SECTIONS[@]} -eq 0 ]; then
    SECTIONS=(overview E1 E1b E2 E2K4 E3 E3T E4)
fi
# eth_proxy gives outbound internet (the W&B upload); sbatch passes the environment on to the jobs
module load eth_proxy
# The env's python directly: conda activate inside --wrap left the jobs on the system python (no torch, Nov 19)
PYTHON=${CONDA_ROOT}/envs/${CONDA_ENVIRONMENT}/bin/python
PYTHON_LIBS=${CONDA_ROOT}/envs/${CONDA_ENVIRONMENT}/lib
FIGURES=${SCRATCH}/report_wandb/figs
OUT=${SCRATCH}/report_wandb/logs
mkdir -p "${OUT}"

for SECTION in "${SECTIONS[@]}"; do
    # W&B files on scratch, not in home (the home inode quota, Nov 6)
    sbatch --job-name="report_${SECTION}" --time=08:00:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G \
        --output="${OUT}/${SECTION}_%j.out" --wrap="
        export OMP_NUM_THREADS=2 WANDB_DIR=${SCRATCH}/report_wandb LD_LIBRARY_PATH=${PYTHON_LIBS}
        cd ${PROJECT_ROOT}
        ${PYTHON} scripts/report_to_wandb.py --section ${SECTION} --root ${SCRATCH} --figures ${FIGURES} --group ${GROUP}"
done
