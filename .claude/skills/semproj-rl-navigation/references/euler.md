# Euler sweeps

## How sweeps run

- **Launcher:** Hydra multirun with `hydra/launcher=euler` (`configs/hydra/launcher/euler.yaml`, submitit). Each job gets 1 CPU, 2 GB and no GPU, with a default `timeout_min: 300`. Timed-out jobs are killed, not resumed.
- **Blocking:** the launcher process blocks until every job finishes. So a script's final `evaluate_metrics.py` call runs after training.
- **Parallel sweeps:** to run several independent multiruns in parallel, start each with `&` and then `wait`.
- **Throughput:** about 16 concurrent jobs.
- **Storage:** `/cluster/scratch/eazevedo/semproj/<GROUP_NAME>`. Scratch is purged periodically, so copy final metric arrays somewhere persistent.

## Cost per run (measured in the E0 pilot, Nov 1)

- **Iteration size:** 256 envs × 256 steps = 65,536 env steps per iteration, capped at `n_iterations=1000` (about 65.5M steps).
- **Per-iteration cost doesn't depend on the room:** 5–8 s (MLP) and 7–10 s (LSTM) on 1 CPU, in two_room, four_room and medium_maze alike.
- **Full run:** ≈ 2.1 h (MLP), ≈ 2.8 h (LSTM). `hydra.launcher.timeout_min=240` is enough; 720 only wastes scheduling priority.
- **Solved rooms stop early:** two_room/four_room with colour off and a random goal stop at 115–220 iterations, 11–26 min.
- **Launcher job:** sbatch the sweep script itself with `--time=24:00:00`, or SLURM's default 1 h kills the launcher and the final `evaluate_metrics.py` never runs.
- **W&B:** train.py logs `Metrics/*` (AUC, success, CR, SPL, nDTW, DTW, NPL) on each run; `scripts/log_metrics_to_wandb.py storage=… timestamp=… wandb.group=…` backfills runs that ended before that existed.

## Script template (copy the header verbatim from an existing script)

```bash
#!/usr/bin/env bash
set -euo pipefail

ETH_USERNAME=eazevedo
PROJECT_NAME=semproj
SCRATCH=/cluster/scratch/${ETH_USERNAME}/${PROJECT_NAME}
CONDA_ENVIRONMENT=semproj
CONDA_ROOT=${HOME}/miniforge3
# ... PROJECT_ROOT resolution, module load eth_proxy, conda activate (see scripts/hard_rooms.sh) ...

GROUP_NAME="e1_place_code_audit"
ROOM_PATHS=("rooms/two_room.json" "rooms/four_room.json")
SEEDS=(100 101 102 103 104)
NOW=$(date +"%Y%m%d_%H%M%S")      # one timestamp for the whole sweep
STORAGE=${SCRATCH}/${GROUP_NAME}

join() { local IFS=","; echo "$*"; }

cd "${PROJECT_ROOT}"
python scripts/train.py --multirun \
    hydra/launcher=euler \
    hydra.launcher.submitit_folder="${STORAGE}/logs/%j" \
    env.room_path="$(join "${ROOM_PATHS[@]}")" \
    seed="$(join "${SEEDS[@]}")" \
    wandb.group="${GROUP_NAME}" \
    storage="${STORAGE}" \
    timestamp="${NOW}" \
    || echo "Some training jobs failed, plotting the seeds that finished" >&2

python scripts/evaluate_metrics.py wandb.group="${GROUP_NAME}" storage="${STORAGE}" timestamp="${NOW}"
```

## Gotchas

- **Timestamps become integers:** Hydra parses `timestamp=20261105_101500` as the integer 20261105101500 (underscores are digit separators), so sweep folders are named without the underscore. Every script uses `str(cfg.timestamp)`, so a sweep matches its own folders, but a hand-made folder name must drop the underscore.
- **zsh doesn't split variables:** `$OVERRIDES` stays one argument in zsh. Run launch snippets with bash, or use bash arrays `"${ARR[@]}"` as the sweep scripts do.
- **Removed keys:** overriding a key that no longer exists (e.g. the removed `env.goal_pos.x`) fails the whole multirun. Check overrides against `configs/` first. A new key needs `+key=value` unless it is declared in a config file.
- **Comma lists:** Hydra treats a comma in a value as a sweep separator. Quote list values: `'observation.masked_channels=[3]'`.
- **Product:** every comma-swept key multiplies the job count. Count jobs before launching (rooms × seeds × each swept key).
- **Combined values:** to vary two settings together rather than as a product (e.g. `max_steps` scaled with corridor length L), run separate multirun invocations.
- **Recurrent constraint:** recurrent runs need `env.num_envs % algorithm.minibatch_size == 0`.
- **Unique storage:** every swept setting must be part of the storage/architecture name in `train.py`, or seeds overwrite each other.
