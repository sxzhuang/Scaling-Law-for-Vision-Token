#!/usr/bin/env bash
# Execute all four stages only when explicitly invoked by the user.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/01_select_fig1_samples.py"
"${SCRIPT_DIR}/02_render_selected_samples.sh"
"${SCRIPT_DIR}/03_run_deepseek_ocr.sh"
"${SCRIPT_DIR}/04_build_eval_results.sh"
