#!/bin/sh

if [ $# -eq 0 ]; then
  PIPELINE_CFG=""
else
  PIPELINE_CFG="$1"
fi

DIR=$(pwd)
OUTPUT_DIR="$DIR/trivia_output"
EVAL_DIR="$DIR/trivia_eval"

mkdir ${EVAL_DIR}
wget -nc "www.stud.fit.vutbr.cz/~ifajcik/r2d2/Trivia/triviaqa-open_test.jsonl" -P "${EVAL_DIR}"

mkdir ${OUTPUT_DIR}

python run_prediction.py "$EVAL_DIR/triviaqa-open_test.jsonl" "$OUTPUT_DIR/predictions.jsonl" "$PIPELINE_CFG" || exit 1

python -m utility.evaluate_predictions --references_path="${EVAL_DIR}/triviaqa-open_test.jsonl" --predictions_path="${OUTPUT_DIR}/predictions.jsonl" || exit 1
