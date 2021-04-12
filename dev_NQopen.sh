#!/bin/sh

if [ $# -eq 0 ]; then
  PIPELINE_CFG=""
else
  PIPELINE_CFG="$1"
fi

DIR=$(pwd)
OUTPUT_DIR="$DIR/nqopen_val_output"
EVAL_DIR="$DIR/nqopen_val_eval"

mkdir ${EVAL_DIR}
wget -nc "www.stud.fit.vutbr.cz/~ifajcik/NQ/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl.zip" -P "${EVAL_DIR}"
unzip -n "$EVAL_DIR/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl.zip" -d "$EVAL_DIR"

mkdir ${OUTPUT_DIR}

python run_prediction.py "$EVAL_DIR/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl" "$OUTPUT_DIR/predictions.jsonl" "$PIPELINE_CFG" || exit 1

python -m utility.evaluate_predictions --references_path="${EVAL_DIR}/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl" --predictions_path="${OUTPUT_DIR}/predictions.jsonl" || exit 1
