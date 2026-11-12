#!/usr/bin/env bash
#
# Run MolCA stage 2 end-to-end with ChEBI-20 as the captioning benchmark:
#   (A) pretrain stage 2 : projector alignment with frozen LM
#                          (paper: 10 epochs)
#                          dataset: PubchemSTM_distill (matches our Stage 1 corpus)
#   (B) fine-tune stage  : LoRA on LM for ChEBI-20 captioning
#                          (paper: 100 epochs)
#                          dataset: ChEBI-20  (train + val + test from .txt)
#
# Stage (B) automatically picks up `last.ckpt` from stage (A).
# stage2.py routes Stage A through Stage2DM (PubchemSTM_distill train, ChEBI val/test
# auto-mapped) and Stage B through Stage2CheBIDM (ChEBI train/val/test) based on
# the --root argument containing "chebi".
#
# Usage:
#   script/run_chebi_full.sh --stage1_path all_checkpoints/<stage1>/epoch=99.ckpt
#
# Common overrides (any flag with a default below):
#   --pretrain_root data/PubchemSTM_distill/
#   --ft_root data/ChEBI-20_data
#   --devices "0,1"
#   --opt_model facebook/galactica-1.3b
#   --pretrain_epochs 10
#   --finetune_epochs 100
#   --run_name my_experiment           (default: chebi_<timestamp>)
#   --peft_config lora_config.json
#   --caption_eval_epoch 5
#   --save_every_n_epochs 5
#   --skip_pretrain                    (skip stage A; requires --pretrain_ckpt)
#   --pretrain_ckpt PATH               (use this ckpt as input to stage B)
#
# Smoke-test (1-2 epochs, tiny batches) before a full run:
#   script/run_chebi_full.sh --stage1_path <ckpt> --pretrain_epochs 1 \
#       --finetune_epochs 2 --caption_eval_epoch 1 --save_every_n_epochs 1 \
#       --extra_args "--limit_train_batches 5 --limit_val_batches 2"
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Resolve a python interpreter. Prefer the project venv so this script works
# even when the caller hasn't activated it (e.g. nohup / background launches).
if [ -n "${PYTHON:-}" ]; then
  :  # caller-provided
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "ERROR: no python interpreter found (set PYTHON= or activate venv)." >&2
  exit 2
fi

# ---- defaults (paper-matching where applicable) ----
PRETRAIN_ROOT="data/PubchemSTM_distill/"   # Stage A: same corpus as Stage 1
FT_ROOT="data/ChEBI-20_data"                # Stage B: ChEBI-20 captioning benchmark
STAGE1_CKPT=""
DEVICES="0,1"
OPT_MODEL="facebook/galactica-1.3b"
PRETRAIN_EPOCHS=10        # paper: pretrain stage 2 = 10 epochs
FINETUNE_EPOCHS=100       # paper: molecule captioning fine-tune = 100 epochs
RUN_NAME="chebi_$(date +%Y%m%d_%H%M%S)"
PEFT_CONFIG="lora_config.json"   # r=16, alpha=32, target q_proj/v_proj/out_proj/fc1/fc2 (Galactica/OPT)
CAPTION_EVAL_EPOCH=5
SAVE_EVERY_N_EPOCHS=5
INFERENCE_BATCH_SIZE=4
BATCH_SIZE=""                # if set, passed as --batch_size
ACCUMULATE_GRAD_BATCHES=""   # if set, passed as --accumulate_grad_batches
PROMPT='[START_I_SMILES]{}[END_I_SMILES]. '
SKIP_PRETRAIN=false
PRETRAIN_CKPT=""
EXTRA_PASSTHROUGH=""         # raw flags forwarded to both stage2.py calls (smoke testing, etc.)

# ---- parse CLI ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1_path)             STAGE1_CKPT="$2"; shift 2 ;;
    --pretrain_root)           PRETRAIN_ROOT="$2"; shift 2 ;;
    --ft_root)                 FT_ROOT="$2"; shift 2 ;;
    --devices)                 DEVICES="$2"; shift 2 ;;
    --opt_model)               OPT_MODEL="$2"; shift 2 ;;
    --pretrain_epochs)         PRETRAIN_EPOCHS="$2"; shift 2 ;;
    --finetune_epochs)         FINETUNE_EPOCHS="$2"; shift 2 ;;
    --run_name)                RUN_NAME="$2"; shift 2 ;;
    --peft_config)             PEFT_CONFIG="$2"; shift 2 ;;
    --caption_eval_epoch)      CAPTION_EVAL_EPOCH="$2"; shift 2 ;;
    --save_every_n_epochs)     SAVE_EVERY_N_EPOCHS="$2"; shift 2 ;;
    --inference_batch_size)    INFERENCE_BATCH_SIZE="$2"; shift 2 ;;
    --batch_size)              BATCH_SIZE="$2"; shift 2 ;;
    --accumulate_grad_batches) ACCUMULATE_GRAD_BATCHES="$2"; shift 2 ;;
    --prompt)                  PROMPT="$2"; shift 2 ;;
    --skip_pretrain)           SKIP_PRETRAIN=true; shift 1 ;;
    --pretrain_ckpt)           PRETRAIN_CKPT="$2"; shift 2 ;;
    --extra_args)              EXTRA_PASSTHROUGH="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,38p' "$0"; exit 0 ;;
    *)
      echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

# ---- validation ----
if ! $SKIP_PRETRAIN && [ -z "$STAGE1_CKPT" ]; then
  echo "ERROR: --stage1_path is required (path to stage 1 checkpoint)." >&2
  echo "       Or pass --skip_pretrain --pretrain_ckpt <path> to skip stage (A)." >&2
  exit 2
fi
if $SKIP_PRETRAIN && [ -z "$PRETRAIN_CKPT" ]; then
  echo "ERROR: --skip_pretrain requires --pretrain_ckpt <path>." >&2
  exit 2
fi
if [ ! -d "$PRETRAIN_ROOT" ]; then
  echo "ERROR: --pretrain_root '$PRETRAIN_ROOT' does not exist." >&2; exit 2
fi
if [ ! -d "$FT_ROOT" ]; then
  echo "ERROR: --ft_root '$FT_ROOT' does not exist." >&2; exit 2
fi
# Sanity: ft root must trigger the Stage2CheBIDM dispatch in stage2.py
if [[ "${FT_ROOT,,}" != *chebi* ]]; then
  echo "ERROR: --ft_root '$FT_ROOT' must contain 'chebi' (case-insensitive) so" >&2
  echo "       stage2.py routes Stage B through Stage2CheBIDM." >&2
  exit 2
fi
if [ ! -f "$PEFT_CONFIG" ]; then
  echo "ERROR: --peft_config '$PEFT_CONFIG' not found." >&2; exit 2
fi

PRETRAIN_NAME="${RUN_NAME}_pretrain"
FT_NAME="${RUN_NAME}_ft"

# Build optional batch-size args once for both stages
EXTRA_ARGS=()
[ -n "$BATCH_SIZE" ]               && EXTRA_ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "$ACCUMULATE_GRAD_BATCHES" ]  && EXTRA_ARGS+=(--accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES")
# Raw passthrough flags (word-split intentionally)
if [ -n "$EXTRA_PASSTHROUGH" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=( $EXTRA_PASSTHROUGH )
fi

export TOKENIZERS_PARALLELISM=false

echo "=================================================================="
echo "  MolCA Stage-2 end-to-end run targeting ChEBI-20"
echo "  Run name      : $RUN_NAME"
echo "  Devices       : $DEVICES"
echo "  Base LM       : $OPT_MODEL"
echo "  Stage A root  : $PRETRAIN_ROOT   (Stage2DM, frozen LM, $PRETRAIN_EPOCHS ep)"
echo "  Stage B root  : $FT_ROOT          (Stage2CheBIDM, LoRA, $FINETUNE_EPOCHS ep)"
echo "  Pretrain name : $PRETRAIN_NAME"
echo "  Fine-tune name: $FT_NAME"
echo "=================================================================="

# ---------- (A) Pretrain Stage 2: projector alignment, frozen LM ----------
if ! $SKIP_PRETRAIN; then
  echo ""
  echo "--- Stage A: Pretrain Stage 2 on $PRETRAIN_ROOT (mode=pretrain, llm_tune=freeze) ---"
  "$PYTHON" stage2.py \
      --root "$PRETRAIN_ROOT" \
      --filename "$PRETRAIN_NAME" \
      --stage1_path "$STAGE1_CKPT" \
      --opt_model "$OPT_MODEL" \
      --devices "$DEVICES" \
      --mode pretrain \
      --max_epochs "$PRETRAIN_EPOCHS" \
      --llm_tune freeze \
      --tune_gnn \
      --prompt "$PROMPT" \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --caption_eval_epoch "$CAPTION_EVAL_EPOCH" \
      --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
      "${EXTRA_ARGS[@]}"
else
  echo ""
  echo "--- Stage A skipped (--skip_pretrain) ---"
fi

# ---------- locate stage A's last checkpoint ----------
if [ -z "$PRETRAIN_CKPT" ]; then
  PRETRAIN_CKPT="all_checkpoints/${PRETRAIN_NAME}/last.ckpt"
fi
if [ ! -f "$PRETRAIN_CKPT" ]; then
  echo "ERROR: Pretrain stage-2 checkpoint not found at: $PRETRAIN_CKPT" >&2
  echo "       Stage A may have failed; check logs above." >&2
  exit 3
fi
echo ""
echo "Pretrain ckpt resolved: $PRETRAIN_CKPT"

# ---------- (B) Fine-tune on ChEBI-20: LoRA on LM, generation ----------
echo ""
echo "--- Stage B: Fine-tune on $FT_ROOT (mode=ft, llm_tune=lora) ---"
"$PYTHON" stage2.py \
    --root "$FT_ROOT" \
    --filename "$FT_NAME" \
    --stage2_path "$PRETRAIN_CKPT" \
    --opt_model "$OPT_MODEL" \
    --devices "$DEVICES" \
    --mode ft \
    --max_epochs "$FINETUNE_EPOCHS" \
    --llm_tune lora \
    --peft_config "$PEFT_CONFIG" \
    --tune_gnn \
    --prompt "$PROMPT" \
    --inference_batch_size "$INFERENCE_BATCH_SIZE" \
    --caption_eval_epoch "$CAPTION_EVAL_EPOCH" \
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=================================================================="
echo "  Done. Checkpoints in:"
echo "    Stage A: all_checkpoints/${PRETRAIN_NAME}/"
echo "    Stage B: all_checkpoints/${FT_NAME}/"
echo "=================================================================="
