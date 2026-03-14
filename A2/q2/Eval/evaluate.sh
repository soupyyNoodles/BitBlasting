#!/usr/bin/env bash
# =============================================================================
# evaluate.sh  —  Forest Fire Spread Evaluator  (COL761 A2 Q2)
# =============================================================================
#
# Usage:
#   bash evaluate.sh <graph_file> <seed_file> <blocked_file> <k> <num_sim> [hops]
#
# Arguments:
#   graph_file    Absolute path to the graph edge-list  (u v p_uv per line)
#   seed_file     Absolute path to the seed-set file    (one node ID per line)
#   blocked_file  Absolute path to the blocked-routes file (u v per line)
#   k             Budget — maximum number of blocked edges allowed
#   num_sim       Number of Monte-Carlo simulation instances
#   hops          (Optional) Limit fire spread to this many hops from the seed
#                 set. Use -1 or omit entirely for unlimited spread (default).
#
# Budget behaviour:
#   - Edges not in the graph and duplicate edges are always rejected.
#   - If the file contains MORE than k valid unique edges:
#       [WARNING] is printed; only the first k (in file order) are evaluated.
#   - If the file contains FEWER than k valid unique edges:
#       Evaluation proceeds; [WARNING] is printed with the results.
#
# Output includes:
#   σ(∅)   expected spread with no blocking
#   σ(R)   expected spread with the blocked edges
#   ρ(R)   reduction ratio  = (σ(∅) − σ(R)) / σ(∅)
#
#   A machine-readable SUMMARY line at the end:
#   SUMMARY  sigma_empty  sigma_R  rho_R  edges_evaluated  k  num_sim  hops
# =============================================================================

set -euo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: bash evaluate.sh <graph_file> <seed_file> <blocked_file> <k> <num_sim> [hops]"
    echo ""
    echo "  hops  (optional) — limit fire spread to this many hops from the seed set."
    echo "                     Pass -1 or omit for unlimited spread (default)."
    exit 1
fi

GRAPH_FILE="$1"
SEED_FILE="$2"
BLOCKED_FILE="$3"
K="$4"
NUM_SIM="$5"
HOPS="${6:--1}"   # default to -1 (unlimited) when not supplied

# Locate evaluate.py in the same directory as this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR="${SCRIPT_DIR}/evaluate.py"

if [ ! -f "$EVALUATOR" ]; then
    echo "[ERROR] evaluate.py not found at: $EVALUATOR"
    exit 1
fi

time python3 "$EVALUATOR" --graph_file "$GRAPH_FILE" --seed_file "$SEED_FILE" --blocked_file "$BLOCKED_FILE" \
    --k "$K" --num_sim "$NUM_SIM" --hops "$HOPS"
