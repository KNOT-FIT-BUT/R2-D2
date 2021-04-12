#!/bin/sh

if [ $# -eq 0 ]; then
  PIPELINE_CFG=""
else
  PIPELINE_CFG="$1"
fi

DIR=$(pwd)
OUTPUT_DIR="$DIR/nqopen_output"
EVAL_DIR="$DIR/nqopen_eval"

mkdir ${EVAL_DIR}
wget -nc https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl -P "${EVAL_DIR}"

mkdir ${OUTPUT_DIR}

python run_prediction.py "$EVAL_DIR/NQ-open.dev.jsonl" "$OUTPUT_DIR/predictions.jsonl" "$PIPELINE_CFG" || exit 1

python -m utility.evaluate_predictions --references_path="${EVAL_DIR}/NQ-open.dev.jsonl" --predictions_path="${OUTPUT_DIR}/predictions.jsonl" || exit 1
