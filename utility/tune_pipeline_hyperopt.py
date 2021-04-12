import copy
import logging
import math
import os
import pickle
import sys
import traceback
import json

import numpy as np
import torch
from hyperopt import hp, fmin, space_eval, Trials, tpe, STATUS_OK, STATUS_FAIL

from configurations.pipeline_configurations import (r2d2_full, r2d2_pruned)
from fit_qa.scripts.common.utility import setup_logging
from fit_qa.scripts.common.evaluate_predictions import evaluate_predictions

import prediction as efficientqa


# dataset without answer
input_path = "/tmp/efficientqa_input/NQ-open.efficientqa.dev.1.1.no-annotations.jsonl"

# efficientQA pipeline predictions
predictions_path = "/tmp/efficientqa_output/tune_pipeline_predictions.jsonl"

# reference dataset
references_path = "/tmp/efficientqa_eval/NQ-open.efficientqa.dev.1.1.jsonl"

# format of path to previous optimization trials
trials_path_format = "efficientqa_pipeline_{}.hyperopt"

# default config of efficientQA pipeline
default_config = r2d2_pruned

# hyperparameters for optimization
reader_batch = hp.choice("reader_batch", [8, 16, 10, 12, 15, 20, 25, 30, 32, 35, 40])

full_index_model_space = {
    # retriever
    #"retriever_model": hp.choice("retriever_model", [
    #    "queryenc_checkpoint_<class 'fit_qa.scripts.models.query_encoder_trainer_wikipassages.QueryEncoderFrameworkWikiPassages'>_Hdist@50_0.80722_2020-10-06_12:23_pcfajcik_PRUNED_HALF.pt.zip",
    #    "queryenc_checkpoint_<class 'fit_qa.scripts.models.query_encoder_trainer_wikipassages.QueryEncoderFrameworkWikiPassages'>_Hdist@50_0.83380_2020-09-26_04:28_supergpu7.fit.vutbr.cz_.pt.zip"
    #    ]),
    #"retriever_top_k": hp.choice("retriever_top_k", [50, 100, 200]),
    #"retriever_top_k": 200,
    # reranker
    #"reranker_top_k": hp.randint("reranker_top_k", 20),
    #"reranker_use_top_k_from_retriever": hp.choice("reranker_use_top_k_from_retriever", [0, reader_batch-2, reader_batch-1]),
    # reader
    #"reader_batch": 20
    "reader_batch": reader_batch
    }

pruned_index_model_space = full_index_model_space


tune_config = {"r2d2_full": (r2d2_full, full_index_model_space),
               "r2d2_pruned": (r2d2_pruned, pruned_index_model_space)}


class EfficientQAPipelineWrapper:
    def __init__(self, input_path, predictions_path, references_path, 
                 default_config):
        self.input_path = input_path
        self.predictions_path = predictions_path
        self.references_path = references_path
        self.default_config = default_config

    def __call__(self, opt_config):
        save_config = efficientqa.config

        config = copy.deepcopy(self.default_config)
        config.update(opt_config)
        efficientqa.config = config
        efficientqa.run_predictions(self.input_path, self.predictions_path)

        metrics = evaluate_predictions(self.references_path, self.predictions_path, 
                                       is_regex=False)

        efficientqa.config = save_config

        return metrics


def objective(self, opt_config, trials):
    """ objective function to optimize (minimize). """

    if len(trials.trials)>1:
        for x in trials.trials[:-1]:
            space_point_index = dict([(key,value[0]) for key,value in x['misc']['vals'].items() if len(value)>0])
            if opt_config == space_eval(model_space, space_point_index):
                loss = x['result']['loss']
                return {'loss': loss, 'status': STATUS_FAIL}
    
    try:
        if "reranker_top_k" in opt_config:
            assert "reranker_use_top_k_from_retriever" not in opt_config
            opt_config["reranker_use_top_k_from_retriever"] = opt_config["reader_batch"] - opt_config["reranker_top_k"]
            assert opt_config["reranker_use_top_k_from_retriever"] >= 0

        metrics = self(opt_config)

        assert "status" not in metrics and "loss" not in metrics
        metrics['status'] = STATUS_OK
        metrics['loss'] = -metrics["accuracy"]

    except BaseException as be:
        logging.error(be)
        logging.error(traceback.format_exc())
        metrics = {'loss': 0, 'status': STATUS_FAIL}

    logging.info("-" * 30)
    logging.info("Config: " + json.dumps(opt_config))
    logging.info("Metrics: " + json.dumps(metrics))
    logging.info("-" * 30)

    return metrics


def run_hyperparam_opt():
    def run_trials(model, obj):
        trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
        max_trials = 5  # initial max_trials. put something small to not have to wait

        try:  # try to load an already saved trials object, and increase the max
            trials = pickle.load(open(trials_path, "rb"))
            logging.info("#" * 30)
            logging.info("#" * 30)
            logging.info("Found saved Trials! Loading...")
            max_trials = len(trials.trials) + trials_step
            logging.info(
                "Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
        except:  # create a new trials object and start searching
            trials = Trials()
            logging.info("Previous trials not found! Creating new trials...")

        best = fmin(fn=lambda params: obj(model, params, trials), space=model_space, algo=tpe.suggest, max_evals=max_trials,
                    trials=trials)
        
        logging.info("Best hyperparam: " + json.dumps(space_eval(model_space, best)))

        # save the trials object
        with open(trials_path, "wb") as f:
            pickle.dump(trials, f)


    wrapper = EfficientQAPipelineWrapper(input_path, predictions_path, 
                                         references_path, default_config)

    # loop indefinitely and stop whenever you like
    while True:
        run_trials(wrapper, objective)


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    
    setup = "r2d2_pruned"

    default_config, model_space = tune_config.get(setup)
    trials_path = trials_path_format.format(setup)

    run_hyperparam_opt()
