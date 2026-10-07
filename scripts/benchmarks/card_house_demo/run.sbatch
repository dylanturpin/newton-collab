#!/bin/bash
#SBATCH --job-name=fpgs-card-house
#SBATCH --partition=a6000-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/data/home/liinbor/src/fpgs-card-house-20261005/job-%j.log
set -eu
cd /data/home/liinbor/src/fpgs-card-house-20261005
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MUJOCO_GL=egl
export PYTHONPATH=/data/home/liinbor/src/fpgs-kitchen-pour-20261005
runtime=/data/home/liinbor/src/fpgs-kitchen-pour-20261005/.venv/bin/python
uv_bin=/data/home/liinbor/tools/uv/uv
nvidia-smi --query-gpu=name,driver_version --format=csv
status=0
run_name=${CARD_RUN:-run-02}
substeps=${CARD_SUBSTEPS:-8}
iterations=${CARD_ITERATIONS:-32}
ball_flags=()
if [ "${CARD_ROLLING:-0}" = "1" ]; then
    ball_flags+=(--rolling-ball)
fi
"$uv_bin" run --no-project --python "$runtime" python simulate.py --output "$run_name" --substeps "$substeps" --iterations "$iterations" --thickness "${CARD_THICKNESS:-0.001}" --angle "${CARD_ANGLE:-20}" --gap "${CARD_GAP:-0.0005}" --impact "${CARD_LAUNCH:-3}" --ball-speed "${CARD_BALL_SPEED:-2.2}" "${ball_flags[@]}" || status=$?
if [ -f "$run_name/result.json" ]; then
    "$uv_bin" run --no-project --python "$runtime" --with mujoco --with pillow --with imageio-ffmpeg --with numpy python render.py --run "$run_name"
fi
exit "$status"
