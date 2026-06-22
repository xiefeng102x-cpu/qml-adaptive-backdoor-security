#!/bin/bash
# ================================================================
# run_sweep.sh — Full parallel sweep for QML adaptive training (MNIST L=12 + BloodMNIST L=8)
# Place at: <repo_root>/
# Launch with: cd <repo_root>
#           nohup bash run_sweep.sh > logs/sweep_master.log 2>&1 &
#
# MAX_JOBS: max concurrent processes; set based on available server cores
#   Recommended: idle cores (nproc output x idle fraction)
#   Conservative default of 8; increase when server load allows
# ================================================================

SCRIPT="training/train_adaptive_qnn_fast.py"
LOGDIR="logs"
PIDFILE="$LOGDIR/pids.txt"
MAX_JOBS=8    # adjust based on nproc and current load

mkdir -p "$LOGDIR"
> "$PIDFILE"

SEEDS=(42 43 44 45 46)
LAMS=(0.0 0.5 1.0 2.0)

# wait for an available job slot (limits concurrency)
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 10
    done
}

# ── MNIST L=12 (5 seeds x 4 lambda_r = 20 tasks) ───────────────────────
echo "=== Launching MNIST L=12 (MAX_JOBS=$MAX_JOBS) ==="
for seed in "${SEEDS[@]}"; do
    for lam in "${LAMS[@]}"; do
        wait_for_slot
        LOG="$LOGDIR/mnist_L12_s${seed}_lam${lam}.log"
        python -u "$SCRIPT" \
            --seed "$seed" \
            --layer 12 \
            --lambda_r "$lam" \
            --dataset mnist \
            --mode single \
            --data-root results/baseline_mnist \
            --out-root  results/training_time_mnist \
            > "$LOG" 2>&1 &
        PID=$!
        echo "$PID  mnist_L12_s${seed}_lam${lam}" >> "$PIDFILE"
        echo "  pid=$PID  MNIST seed=$seed L=12 λ=$lam"
    done
done

# ── BloodMNIST L=8 (5 seeds x 4 lambda_r = 20 tasks) ──────────────────
echo ""
echo "=== Launching BloodMNIST L=8 (MAX_JOBS=$MAX_JOBS) ==="
for seed in "${SEEDS[@]}"; do
    for lam in "${LAMS[@]}"; do
        wait_for_slot
        LOG="$LOGDIR/blood_L8_s${seed}_lam${lam}.log"
        python -u "$SCRIPT" \
            --seed "$seed" \
            --layer 8 \
            --lambda_r "$lam" \
            --dataset bloodmnist \
            --mode single \
            --data-root results/baseline_bloodmnist \
            --out-root  results/training_time_bloodmnist \
            > "$LOG" 2>&1 &
        PID=$!
        echo "$PID  blood_L8_s${seed}_lam${lam}" >> "$PIDFILE"
        echo "  pid=$PID  BloodMNIST seed=$seed L=8 λ=$lam"
    done
done

# wait for all background jobs to finish
echo ""
echo "All tasks queued, waiting for completion..."
wait
echo "================================================================"
echo "All done!"
echo "  MNIST completed:  $(ls results/training_time_mnist/result_seed=*_L12_*.json 2>/dev/null | wc -l) / 20"
echo "  BloodMNIST completed:  $(ls results/training_time_bloodmnist/result_seed=*_L8_*.json 2>/dev/null | wc -l) / 20"
echo "================================================================"

