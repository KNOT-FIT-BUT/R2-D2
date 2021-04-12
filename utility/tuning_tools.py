import logging
import os
import random
from typing import List, AnyStr

import jsonlines
import torch

from scalingqa.common.utility import eval_utils
from scalingqa.retriever.datasets.openQA_wikipassages import OpenQA_WikiPassages
from utility.evaluate_predictions import exact_match_score
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utility.utility import argmax
import numpy as np


def log_softmax(l):
    x = np.array(l)
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum()).tolist()


def load_ranker_scores(questions, ext_passages, ranker_file, logsoftmax_scores=True):
    with jsonlines.open(ranker_file) as rf:
        ranked_predictions = list(rf)
    ranker_scores = []
    for idx, ranked_p in enumerate(ranked_predictions):
        try:
            q_idx = questions.index(ranked_p['question'])
        except ValueError:
            continue  # answer was not in top-K predictions

        answer_passages = ext_passages[q_idx]
        if logsoftmax_scores:
            ranked_p['predicted_scores'] = log_softmax(ranked_p['predicted_scores'])
        rs = [ranked_p['predicted_scores'][ranked_p['predicted_indices'].index(passage_idx)]
              for passage_idx in answer_passages]
        ranker_scores.append(rs)
    return ranker_scores


def load_ext_reader_scores(extractive_reader_outfile, gt_file=None):
    if gt_file is not None:
        correct = load_correct(gt_file)

    with jsonlines.open(extractive_reader_outfile) as reader_outputs:
        extractive_reader_predictions = list(reader_outputs)

    questions, gt_answers, proposed_answers, ext_scores, ext_passages, labels = [], [], [], [], [], []
    for e in extractive_reader_predictions:
        if gt_file is not None:
            correct_answer_list = correct[e['raw_question']]

            # Get ground truth rank, if there is gt
            gt_rank = -1
            for i, a in enumerate(e['answers']):
                if eval_utils.metric_max_over_ground_truths(
                        metric_fn=exact_match_score, prediction=a, ground_truths=correct_answer_list):
                    gt_rank = i
                    break
            if gt_rank < 0:
                continue

            gt_answers.append(correct_answer_list)
            labels.append(gt_rank)

        questions.append(e['raw_question'])
        proposed_answers.append(e["answers"])

        # Get ranker and reranker scores for each answer's passage
        ext_passages.append(e['passages'])
        ext_scores.append(e['reader_scores'])

    if gt_file is not None:
        validation_metadata = {
            "questions": questions,
            "proposed_answers": proposed_answers,
            "gt_answers": gt_answers
        }
        return ext_scores, ext_passages, validation_metadata, labels
    else:
        metadata = {
            "questions": questions,
            "proposed_answers": proposed_answers,
        }
        return ext_scores, ext_passages, metadata


def load_ext_abs_score_data(extractive_output, abstractive_output, gt_file=None):
    if gt_file is not None:
        correct = load_correct(gt_file)

    questions, gt_answers, proposed_answers, scores, labels, best_span_indices = [], [], [], [], [], []
    with jsonlines.open(extractive_output) as span_predictions, jsonlines.open(abstractive_output) as abs_predictions:
        for ext, abs in zip(span_predictions, abs_predictions):
            assert abs['raw_question'] == ext['raw_question']
            ext_pred, ext_score = ext['answers'][argmax(ext['reader_scores'])], max(ext['reader_scores'])
            best_span_index = argmax(ext['reader_scores'])

            abs_pred, abs_score = abs['answers'][argmax(abs['reader_scores'])], max(abs['reader_scores'])

            if gt_file is not None:
                correct_answer_list = correct[abs['raw_question']]
                is_ext_correct = eval_utils.metric_max_over_ground_truths(
                    metric_fn=exact_match_score, prediction=ext_pred, ground_truths=correct_answer_list)
                is_abs_correct = eval_utils.metric_max_over_ground_truths(
                    metric_fn=exact_match_score, prediction=abs_pred, ground_truths=correct_answer_list)
                if is_ext_correct == is_abs_correct:
                    continue

                # extractive is class 0, abstractive class 1
                labels.append(int(is_abs_correct))
                gt_answers.append(correct_answer_list)

            best_span_indices.append(best_span_index)
            questions.append(abs['raw_question'])
            scores.append([ext_score, abs_score])
            proposed_answers.append([ext_pred, abs_pred])
    assert len(questions) == len(scores) == len(proposed_answers)
    if gt_file is not None:
        validation_metadata = {
            "questions": questions,
            "proposed_answers": proposed_answers,
            "gt_answers": gt_answers,
            "ext_best_span_idx": best_span_indices
        }
        return scores, validation_metadata, labels
    else:
        metadata = {
            "questions": questions,
            "proposed_answers": proposed_answers,
            "ext_best_span_idx": best_span_indices
        }
        return scores, metadata


def load_correct(gt_file):
    with jsonlines.open(gt_file, mode="r") as reader:
        correct = dict((OpenQA_WikiPassages.get_qa_from_example(e) for e in reader))
    return correct


def load_genreader_scores(questions, generative_reader_outfile):
    with jsonlines.open(generative_reader_outfile) as generative_reader_predictions:
        scores = []
        for e in generative_reader_predictions:
            if e['raw_question'] not in questions:
                continue  # answer was not in top-K predictions
            scores.append(e['reader_scores'])
    return scores


def load_pipeline_data(outputs: List[AnyStr], aggregation_config, gt_file=None):
    ext_reader_data = load_ext_reader_scores(outputs["reader_output"], gt_file)
    if gt_file is not None:
        ext_scores, ext_passages, metadata, labels = ext_reader_data
    else:
        ext_scores, ext_passages, metadata = ext_reader_data

    questions = metadata["questions"]
    if not "extractive_reader" in aggregation_config["components"]:
        pipeline_data = []
    else:
        pipeline_data = [ext_scores]
    if "retriever" in aggregation_config["components"]:
        pipeline_data.append(load_ranker_scores(questions, ext_passages, outputs["retriever_output"]))
    if "reranker" in aggregation_config["components"]:
        pipeline_data.append(load_ranker_scores(questions, ext_passages, outputs["passage_reranker_output"]))
    if "generative_reader" in aggregation_config["components"]:
        pipeline_data.append(load_genreader_scores(questions, outputs["answer_reranker_output"]))

    assert all(len(i) == len(pipeline_data[0]) for i in pipeline_data)
    pipeline_data = [list(a) for a in zip(*pipeline_data)]  # transpose values
    if gt_file is not None:
        return pipeline_data, metadata, labels
    else:
        return pipeline_data, metadata


class ConstrainedLR(torch.nn.Module):
    def __init__(self, param_count):
        super().__init__()
        self.linear = torch.nn.Linear(param_count, 1)

    def forward(self, X):
        return self.linear(X).squeeze_(-1)


def evaluate(preds, gts):
    assert len(preds) == len(gts)
    hits = 0
    for q, a in preds.items():
        gt_answers = gts[q]
        hits += int(eval_utils.metric_max_over_ground_truths(
            metric_fn=exact_match_score, prediction=a, ground_truths=gt_answers))
    acc = hits * 100. / len(preds)
    return acc


def evaluate_aggregated_model(validation_logits, validation_metadata):
    fused_predictions = aggregated_predict(validation_logits, validation_metadata)

    ground_truth = dict(zip(validation_metadata["questions"], validation_metadata["gt_answers"]))

    return evaluate(fused_predictions, ground_truth)


def aggregated_predict(logits, metadata):
    fused_predictions = dict()
    assert all(logits.shape[0] == len(v) for v in metadata.values())
    for fused_scores, question, topk_answer_list in zip(logits.tolist(),
                                                        metadata["questions"],
                                                        metadata["proposed_answers"]):
        fused_predictions[question] = topk_answer_list[argmax(fused_scores)]
    return fused_predictions


def ext_abs_predict(logits, metadata):
    fused_predictions = dict()
    assert all(logits.shape[0] == len(v) for v in metadata.values())
    for decision_score, question, ext_or_gen_answer in zip(logits.tolist(),
                                                           metadata["questions"],
                                                           metadata["proposed_answers"]):
        # extractive is class 0, abstractive class 1
        fused_predictions[question] = ext_or_gen_answer[int(decision_score > 0.)]
    return fused_predictions


def run_training(training_data, training_labels, validation_data, validation_metadata, evaluate_model_cb):
    STEPS = 100
    model = ConstrainedLR(training_data.shape[-1])

    model = model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.3)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=10,
        num_training_steps=STEPS
    )

    best_acc = 0
    best_r = None
    iterator = range(STEPS)
    for i in iterator:
        logits = model(training_data)
        loss_f = F.cross_entropy if len(logits.shape) > 1 else F.binary_cross_entropy_with_logits
        l_list = loss_f(logits, training_labels, reduction='none')
        l = l_list.mean()
        l.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()
        if i % 1 == 0:
            r = {k: v.tolist() for k, v in dict(model.linear.named_parameters()).items()}
            with torch.no_grad():
                model.eval()
                acc = evaluate_model_cb(model(validation_data), validation_metadata)
                model.train()
            if acc > best_acc:
                best_acc = acc
                best_r = r
    return best_acc, best_r


def tune_ext_abs_fusion_parameters(extractive_output, abstractive_output, gt_file, trials=50):
    data, validation_metadata, labels = load_ext_abs_score_data(extractive_output, abstractive_output, gt_file)
    data, labels = torch.FloatTensor(data), torch.FloatTensor(labels)
    training_data = validation_data = data
    training_labels = labels
    aggregation_type = 'extractive-abstractive prediction fusion'
    return fit_logistic_regression(training_data, training_labels, validation_data, validation_metadata,
                                   aggregation_type, trials, evaluate_extabs_model)


def evaluate_extabs_model(validation_logits, validation_metadata):
    fused_predictions = ext_abs_predict(validation_logits, validation_metadata)
    ground_truth = dict(zip(validation_metadata["questions"], validation_metadata["gt_answers"]))

    return evaluate(fused_predictions, ground_truth)


def tune_score_aggregation_parameters(outputs: List[AnyStr], aggregation_config, gt_file, trials=50,
                                      use_split_point=False):
    data, validation_metadata, labels = load_pipeline_data(outputs, aggregation_config, gt_file)
    if use_split_point:
        index_shuf = list(range(len(data)))
        random.Random(1234).shuffle(index_shuf)
        shuffled_data = [data[i] for i in index_shuf]
        shuffled_labels = [labels[i] for i in index_shuf]
        validation_metadata = {k: [v[i] for i in index_shuf] for k, v in validation_metadata.items()}

        data, labels = torch.FloatTensor(shuffled_data).transpose(-1, -2), torch.LongTensor(shuffled_labels)
        split_point = round(4 / 5 * len(data))
        training_data, validation_data = data[:split_point], data[split_point:]
        training_labels = labels[:split_point]
        validation_metadata = {k: v[split_point:] for k, v in validation_metadata.items()}
    else:
        data, labels = torch.FloatTensor(data).transpose(-1, -2), torch.LongTensor(labels)
        training_data = validation_data = data
        training_labels = labels

    aggregation_type = 'component score aggregation'
    return fit_logistic_regression(training_data, training_labels, validation_data, validation_metadata,
                                   aggregation_type, trials, evaluate_aggregated_model)


def fit_logistic_regression(training_data, training_labels, validation_data, validation_metadata, aggregation_type,
                            trials, evaluate_model_cb):
    logging.info(f"Running parameter tuning for: {aggregation_type}")
    logging.info(f"Training data: {len(training_data)}, Validation data: {len(validation_data)}")
    total_best_acc, total_best_r = 0., None
    for trial_i in tqdm(range(trials), total=trials):
        acc, r = run_training(training_data, training_labels,
                              validation_data, validation_metadata,
                              evaluate_model_cb)
        if acc > total_best_acc:
            total_best_acc = acc
            total_best_r = r
            logging.info("TRIAL" + str(trial_i))
            logging.info("BEST_ACC " + str(total_best_acc))
            logging.info("BEST_R " + str(total_best_r))
    logging.info(f"Found parameters for: {aggregation_type}\n"
                 f"{total_best_r}")
    return total_best_r
