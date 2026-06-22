#!/bin/bash
# ============================================================
# run_baseline_mnist.sh
# Full baseline grid sweep for MNIST (5-qubit, amplitude encoding)
# Seeds 42-46 × layers 3-12 × non-target classes × poison ratios
#
# Usage:
#   bash training/run_baseline_mnist.sh
#   bash training/run_baseline_mnist.sh --max-jobs 16
#
# Output: results/baseline_mnist/seed_*/layer_*/t7_vs_t*/pr_*/
# Estimated runtime: ~12-24 h on 8 CPU cores
# ============================================================

set -euo pipefail

MAX_JOBS=8
for arg in "$@"; do
  case $arg in
    --max-jobs=*) MAX_JOBS="${arg#*=}" ;;
    --max-jobs)   shift; MAX_JOBS="$1" ;;
  esac
done

SCRIPT="training/scb_mnist_baseline.py"
OUTBASE="results/baseline_mnist"
LOGDIR="logs/baseline_mnist"
mkdir -p "$LOGDIR"

SEEDS=(42 43 44 45 46)
LAYERS=(3 4 5 6 7 8 9 10 11 12)
NON_TARGETS=(0 1 2 3 4 5 6 8 9)   # all classes except target (7)
POISON_RATIOS=(0.1 0.2 0.3 0.4 0.5)

total=$(( ${#SEEDS[@]} * ${#LAYERS[@]} * ${#NON_TARGETS[@]} * ${#POISON_RATIOS[@]} ))
echo "Total tasks: $total  (MAX_JOBS=$MAX_JOBS)"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 5; done
}

done_count=0
skip_count=0

for seed in "${SEEDS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for nt in "${NON_TARGETS[@]}"; do
      for pr in "${POISON_RATIOS[@]}"; do
        outdir="$OUTBASE/seed_${seed}/layer_${layer}_grid/t7_vs_t${nt}/pr_${pr}"
        logfile="$LOGDIR/s${seed}_L${layer}_t${nt}_pr${pr}.log"

        if [ -f "${outdir}/_DONE" ]; then
          (( skip_count++ )) || true
          continue
        fi

        wait_for_slot
        (( done_count++ )) || true
        mkdir -p "$outdir"

        python "$SCRIPT" \
          --random-seed "$seed" \
          --n-seeds 1 \
          --target-class 7 \
          --non-target-class "$nt" \
          --n-qubits 5 \
          --n-layers "$layer" \
          --epsilon 0.8 \
          --poison-ratios "$pr" \
          --apply-pca \
          --standardize \
          --output-dir "$outdir" \
          > "$logfile" 2>&1 \
          && touch "${outdir}/_DONE" \
          || echo "FAILED: seed=$seed L=$layer nt=$nt pr=$pr" &

        echo "  launched [${done_count}/${total}] seed=$seed L=$layer t7_vs_t${nt} pr=$pr"
      done
    done
  done
done

wait
echo "Done. Launched: $done_count  Skipped (already complete): $skip_count"
