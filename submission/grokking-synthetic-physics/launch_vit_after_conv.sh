#!/bin/bash
# Auto-launch ViT sweep after all ConvNet PIDs finish.
# Usage: nohup bash launch_vit_after_conv.sh > logs/vit_launcher.log 2>&1 &

CONV_PIDS=(747922 747923 747924 747925 747926 747927 747928)
PROJECT_DIR="/data/fl21/CS552_SP26_FinalProject/grokking-synthetic-physics"
LOG="$PROJECT_DIR/logs/vit_launcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "ViT launcher started. Waiting for ConvNet PIDs: ${CONV_PIDS[*]}"

while true; do
    running=()
    for pid in "${CONV_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && running+=("$pid")
    done

    if [ ${#running[@]} -eq 0 ]; then
        log "All ConvNet runs finished."
        break
    fi

    log "Still running (${#running[@]}/7): ${running[*]}"
    sleep 300   # poll every 5 minutes
done

# ── Launch ViT sweep ──────────────────────────────────────────────
source /home/fl21/miniconda3/etc/profile.d/conda.sh && conda activate grokking
cd "$PROJECT_DIR"

dims=(2 4 8 16 32 64 128)
for i in 0 1 2 3 4 5 6; do
    dim=${dims[$i]}
    CUDA_VISIBLE_DEVICES=$i nohup python train.py \
        --model vit --latent_dim "$dim" --epochs 2000 \
        --h5_path pendulum_data_v3.h5 --design_csv episode_design.csv \
        --save_dir "runs/vit_dim${dim}_v2" \
        --keep_checkpoints 3 --save_every 50 \
        --min_epochs 200 --zstd_patience 50 --zstd_threshold 0.01 \
        > "logs/vit_dim${dim}_v2.log" 2>&1 &
    log "Launched vit dim=${dim} on GPU $i (PID $!)"
done

log "All ViT runs launched."
