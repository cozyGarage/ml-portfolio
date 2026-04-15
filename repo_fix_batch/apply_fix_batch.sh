#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: bash apply_fix_batch.sh /path/to/ml-portfolio"
  exit 1
fi
REPO_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -av --exclude '.git' --exclude 'apply_fix_batch.sh' --exclude 'apply_fix_batch.ps1' --exclude 'APPLY_FIXES.md' --exclude 'REPO_AUDIT_AND_FIX_PLAN.md' "$SCRIPT_DIR/" "$REPO_PATH/"
echo "Applied fix batch to $REPO_PATH"
