#!/usr/bin/env bash
# Run H2 experiments: LapPE baseline vs. no-PE vs. LapPE + GraphStatsSE.
#
# After completion, collect results with:
#   python scripts/collect_h2_results.py

set -euo pipefail

REPEAT_SMALL="${REPEAT_SMALL:-10}"

cd "$(dirname "$0")/.."

echo "========================================================"
echo "  H2 GraphStats Ablation — GraphGPS"
echo "  Seeds (small datasets): $REPEAT_SMALL"
echo "  Working directory:    $(pwd)"
echo "========================================================"

run_exp() {
    local cfg_file="$1"
    local label="$2"
    local repeat="${3:-$REPEAT_SMALL}"
    echo ""
    echo "--- [$label] $cfg_file  (seeds: $repeat) ---"
    python main.py \
        --cfg "$cfg_file" \
        --repeat "$repeat" \
        wandb.use False
    echo "--- done: $cfg_file ---"
}

run_exp "configs/GPS/mutag-GPS.yaml" "H2 MUTAG LapPE"
run_exp "configs/GPS/enzymes-GPS.yaml" "H2 ENZYMES LapPE"
run_exp "configs/GPS/nci1-GPS.yaml" "H2 NCI1 LapPE"

run_exp "configs/GPS/mutag-GPS-noPE.yaml" "H2 MUTAG no-PE"
run_exp "configs/GPS/enzymes-GPS-noPE.yaml" "H2 ENZYMES no-PE"
run_exp "configs/GPS/nci1-GPS-noPE.yaml" "H2 NCI1 no-PE"

run_exp "configs/GPS/mutag-GPS+GraphStatsSE.yaml" "H2 MUTAG LapPE+GraphStatsSE"
run_exp "configs/GPS/enzymes-GPS+GraphStatsSE.yaml" "H2 ENZYMES LapPE+GraphStatsSE"
run_exp "configs/GPS/nci1-GPS+GraphStatsSE.yaml" "H2 NCI1 LapPE+GraphStatsSE"

echo ""
echo "========================================================"
echo "  All H2 experiments complete."
echo "  Collect results: python scripts/collect_h2_results.py"
echo "========================================================"