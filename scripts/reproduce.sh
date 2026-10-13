#!/usr/bin/env bash
# The paper's Qwen3-Omni experiments, stage by stage.
#
#   ALGONAUTS_ROOT=/path/to/algonauts_2025.competitors scripts/reproduce.sh data
#   scripts/reproduce.sh init
#   scripts/reproduce.sh main
#
# Every run trains on one 80 GB GPU (~4.8 h with LoRA, ~1.7 h readout-only),
# then is re-fitted in closed form and scored on Friends season 6.
set -euo pipefail

DATA=${DATA:-data}
RUNS=${RUNS:-runs}
INIT=$DATA/readout_init_L32.npz

train_and_refit() {            # name [train flags...]
    local name=$1; shift
    python -m brain_lora.cli.train --data-dir "$DATA" --readout-init "${RUN_INIT:-$INIT}" \
        --run-dir "$RUNS/$name" "$@"
    python -m brain_lora.cli.refit "$RUNS/$name"
}

fit_init() {                   # context lead (s), output
    python -m brain_lora.cli.extract_features --data-dir "$DATA" --context-lead-s "$1" \
        --out "$DATA/features_L32_lead$1.h5"
    python -m brain_lora.cli.fit_readout_init --data-dir "$DATA" \
        --features "$DATA/features_L32_lead$1.h5" --out "$2"
}

case "${1:-}" in
data)        # media cache, window manifests, z-scored fMRI
    python -m brain_lora.cli.prepare_data --algonauts-root "$ALGONAUTS_ROOT" --out "$DATA" ;;
init)        # readout initialisation from frozen features (30 s context lead)
    fit_init 30 "$INIT" ;;
main)        # main results: {frozen, LoRA} x {linear, cross-attention}
    train_and_refit lora_linear
    train_and_refit lora_cross_attn --readout cross_attn
    train_and_refit frozen_linear --lora-lr 0
    train_and_refit frozen_cross_attn --readout cross_attn --lora-lr 0 ;;
ablations)   # one change at a time from LoRA + linear
    train_and_refit rank32 --rank 32
    train_and_refit gated_mlp --readout gated_mlp
    train_and_refit no_audio --no-audio ;;
lead)        # context-lead ladder (0 s starts from an init fitted without lead)
    for lead in 30 90 120; do
        train_and_refit "lead$lead" --context-lead-s "$lead"
    done
    fit_init 0 "$DATA/readout_init_L32_lead0.npz"
    RUN_INIT=$DATA/readout_init_L32_lead0.npz train_and_refit lead0 --context-lead-s 0 ;;
seeds)       # seed spread: LoRA + linear (15 seeds) and LoRA + cross-attention (3)
    for s in $(seq 1 14); do train_and_refit "lora_linear_s$s" --seed "$s"; done
    for s in 1 2; do train_and_refit "lora_cross_attn_s$s" --readout cross_attn --seed "$s"; done ;;
attribution) # modality attribution, adapted and frozen cross-attention models
    for run in lora_cross_attn frozen_cross_attn; do
        out=$RUNS/$run/attribution
        python -m brain_lora.analysis.attribution baseline --run-dir "$RUNS/$run" --out "$out"
        python -m brain_lora.analysis.attribution run --run-dir "$RUNS/$run" --out "$out"
        python -m brain_lora.analysis.attribution report --out "$out"
    done ;;
compare)     # paired block-bootstrap comparison, e.g. the adapter's gain
    python -m brain_lora.analysis.compare "$RUNS/frozen_linear" "$RUNS/lora_linear" \
        --data-dir "$DATA" ;;
*)
    echo "usage: $0 {data|init|main|ablations|lead|seeds|attribution|compare}" >&2
    exit 1 ;;
esac
