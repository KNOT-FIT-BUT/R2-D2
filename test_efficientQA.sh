#!/bin/sh

if [ $# -eq 0 ]; then
  PIPELINE_CFG=""
else
  PIPELINE_CFG="$1"
fi

DIR=$(pwd)
OUTPUT_DIR="$DIR/efficientqa_output"
EVAL_DIR="$DIR/efficientqa_eval"

mkdir ${EVAL_DIR}
wget -nc https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.efficientqa.dev.1.1.jsonl -P "${EVAL_DIR}"

mkdir ${OUTPUT_DIR}

python run_prediction.py "$EVAL_DIR/NQ-open.efficientqa.dev.1.1.jsonl" "$OUTPUT_DIR/predictions.jsonl" "$PIPELINE_CFG" || exit 1

python -m utility.evaluate_predictions --references_path="${EVAL_DIR}/NQ-open.efficientqa.dev.1.1.jsonl" --predictions_path="${OUTPUT_DIR}/predictions.jsonl" || exit 1
