"""
This file contains tools for producing predictions

python run_predictions_[full|pruned].py <input_file> <output_file>

<input_file> format
{"question": "who won the women's australian open 2018"}
{"question": "kuchipudi is a dance form of which state"}
{"question": "who did stephen amell play in private practice"}
{"question": "who created the chamber of secrets in harry potter"}

<output_file> format
{"question": "who won the women's australian open 2018", "prediction": "Caroline Wozniacki"}
{"question": "kuchipudi is a dance form of which state", "prediction": "Tamil Nadu"}
{"question": "who did stephen amell play in private practice", "prediction": "a pedestrian"}
{"question": "who created the chamber of secrets in harry potter", "prediction": "the Heir of Salazar Slytherin"}

"""
import json
import logging
import os

import h5py
import scalingqa.reranker.run_reranker as reranker_module
import torch
import torch.nn.functional as F
from jsonlines import jsonlines
from scalingqa.common.utility.utility import mkdir
from scalingqa.extractivereader.answer_extractor import AnswerExtractor
from scalingqa.extractivereader.datasets.pass_database import PassDatabase
from scalingqa.extractivereader.datasets.reader_dataset import ReaderDataset
from scalingqa.extractivereader.models.reader import Reader
from scalingqa.extractivereader.utils.checkpoint import Checkpoint
from scalingqa.generative_reader.training.generative_reader_trainer_fid import FIDTrainer
from scalingqa.retriever.models.lrm_encoder import LRM_encoder
from scalingqa.retriever.query_encoder_trainer_wikipassages import QueryEncoderFrameworkWikiPassages
from tqdm import tqdm
from transformers import AutoTokenizer

from configurations.pipeline_configurations import DEFAULT_CFG_FILE
from utility.evaluate_predictions import evaluate_predictions
from utility.tuning_tools import tune_score_aggregation_parameters, load_pipeline_data, tune_ext_abs_fusion_parameters, \
    load_ext_abs_score_data
from utility.utility import lazy_unzip, argmax, download_item


def create_filename(prefix, name, suffix="jsonl"):
    return os.path.join(config["pipeline_cache_dir"],
                        os.path.basename(prefix) + f"_{name}.{suffix}")


def get_database_path():
    directory = config["index"]["database"]["directory"]
    filename = config["index"]["database"]["name"]
    return os.path.join(directory, filename)


def is_component_active(component, configuration=None):
    return component in configuration and configuration[component].get("active", "false") != "false"


def is_tuning_needed(parameters):
    return parameters == "tune_please!"


def run_retriever(infile, outfile):
    def get_retriever_model(device):
        retriever_cfg = config["retriever"]
        model_path = os.path.join(retriever_cfg["directory"], retriever_cfg["name"])

        logging.info(f"Loading retriever model from {model_path}")

        download_item(retriever_cfg["url"], model_path)
        if model_path.endswith("zip"):
            lazy_unzip(model_path)
            model_path = model_path[:-len(".zip")]
        model_dict = torch.load(model_path)
        model_dict["config"]["model_cache_dir"] = config["transformers_cache"]
        m = LRM_encoder(model_dict["config"], do_not_download_weights=True)
        m.load_state_dict(model_dict["state_dict"])
        return m.float().to(device)  # make sure 32-bit p

    def get_index():
        emb_cfg = config["index"]["passage_embeddings"]
        passage_embeddings_path = os.path.join(emb_cfg["directory"],
                                               emb_cfg["name"])
        logging.info(f"Loading wiki embeddings from {passage_embeddings_path}")
        if "url" in emb_cfg:
            download_item(emb_cfg["url"], passage_embeddings_path)
        if passage_embeddings_path.endswith(".zip"):
            lazy_unzip(passage_embeddings_path)
            h5p_tensor = h5py.File(passage_embeddings_path[:-len(".zip")], 'r')['data'][()]
        else:
            h5p_tensor = h5py.File(passage_embeddings_path, 'r')['data'][()]
        passage_embeddings = torch.FloatTensor(h5p_tensor)
        del h5p_tensor
        return passage_embeddings

    device = config["device"]
    logging.info("Using device: " + str(device)
                 + "\n - device name: " + torch.cuda.get_device_name(device=device)
                 + "\n - total_memory: " + str(torch.cuda.get_device_properties(device).total_memory / 1e6) + " GB")

    ret_model = get_retriever_model(device)
    index = get_index()
    logging.info(f"Loaded passage embeddings of size: {index.shape}")

    QueryEncoderFrameworkWikiPassages.predict(
        infile=infile,
        outfile=outfile,
        model=ret_model,
        passage_embeddings=index,
        config={
            "parallelize_dot": False,
            "emb_on_gpu": False,
            "data_cache_dir": config["pipeline_cache_dir"],
            "transformers_cache": config["transformers_cache"],
            "batch_size": 32,
            "K": config["retriever"]["top_k"]
        },
        device=device
    )

    # For next steps database will be needed, prepare it here
    db_cfg = config["index"]["database"]
    db_path = os.path.join(db_cfg["directory"], db_cfg["name"])
    logging.info(f"Preparing database into {db_path}")
    if "url" in db_cfg:
        download_item(db_cfg["url"], db_path)
    if db_path.endswith(".zip"):
        lazy_unzip(db_path)
        db_cfg["name"] = db_cfg["name"][:-len(".zip")]


def run_reranker(retrieval_output, reranker_output):
    top_k = config["retriever"]["top_k"]
    reranker_cfg = config["reranker"]

    model_path = os.path.join(reranker_cfg["directory"], reranker_cfg["name"])
    download_item(reranker_cfg["url"], model_path)
    if model_path.endswith(".zip"):
        lazy_unzip(model_path)
        model_path = model_path[:-len(".zip")]

    db_path = get_database_path()
    if db_path.endswith(".zip"):
        db_path = db_path[:-len(".zip")]

    reranker_module.run_reranker(
        infile=retrieval_output,
        outfile=reranker_output,
        database=db_path,
        reranker_model=model_path,
        k_top=top_k,
        cache_dir=config["transformers_cache"],
        batch_size=reranker_cfg["config"]["batch_size"])


def run_reader_generative(checkpoint, reader_output, ranked_output):
    db_path = os.path.join(config["index"]["database"]["directory"], config["index"]["database"]["name"])
    if db_path.endswith(".zip"):
        db_path = db_path[:-len(".zip")]
    generative_reader_config = checkpoint["config"].custom_config
    generative_reader_config.update({
        "transformers_cache": config["transformers_cache"],
        "data_cache_dir": config["pipeline_cache_dir"],
        "pass_database": db_path,
    })
    FIDTrainer.predict(ranked_output, reader_output, checkpoint, generative_reader_config,
                       config["device"])


def run_generative_reader_reranking(extractive_reader_input, extractive_reader_output, reranking_output, gt_file=None):
    # assume db was already unzipped
    db_path = os.path.join(config["index"]["database"]["directory"], config["index"]["database"]["name"])
    if db_path.endswith(".zip"):
        db_path = db_path[:-len(".zip")]

    checkpoint = get_reader_ckpt(config["reader"]["generative"])

    generative_reader_config = checkpoint["config"].custom_config
    generative_reader_config.update({
        "transformers_cache": config["transformers_cache"],
        "data_cache_dir": config["pipeline_cache_dir"],
        "pass_database": db_path,
    })
    FIDTrainer.rerank(extractive_reader_input, reranking_output, extractive_reader_output,
                      checkpoint, generative_reader_config, config["device"], gt_file=gt_file)


def get_reader_ckpt(reader_cfg=None):
    if reader_cfg is None:
        reader_cfg = config["reader"][config["reader"]['active']]
    reader_path = os.path.join(reader_cfg["directory"], reader_cfg["name"])

    download_item(reader_cfg["url"], reader_path)
    if reader_path.endswith("zip"):
        lazy_unzip(reader_path)
        reader_path = reader_path[:-len(".zip")]

    logging.info(f"Loading reader model from {reader_path}")
    return torch.load(reader_path)


def run_reader(ranked_output, reader_output, reader_type=None):
    active = reader_type if reader_type is not None else config["reader"]["active"]
    reader_ckpt = get_reader_ckpt(config["reader"][active])
    if "generative" == active:
        run_reader_generative(reader_ckpt, reader_output, ranked_output)
    elif "extractive" == active:
        run_reader_extractive(reader_ckpt, reader_output, ranked_output)
    else:
        raise ValueError(f'Unknown active reader {config["reader"]["active"]}')


def run_reader_extractive(checkpointDict, reader_output, reranker_output):
    ext_reader_cfg = config["reader"]["extractive"]["config"]
    cache_dir = config["transformers_cache"]

    checkpointDict["config"]["cache"] = cache_dir  # overwrite the old loaded cache path
    model = Reader(checkpointDict["config"], initPretrainedWeights=False)
    Checkpoint.loadModel(model, checkpointDict, config["device"])

    if "multi_gpu" in ext_reader_cfg and ext_reader_cfg["multi_gpu"] and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logging.info("DataParallel active!")

    extractor = AnswerExtractor(model, config["device"])
    extractor.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpointDict["config"]['tokenizer_type'],
                                              cache_dir=cache_dir, use_fast=True)
    database = get_database_path()
    database = PassDatabase(database)
    with ReaderDataset(reranker_output, tokenizer, database,
                       ext_reader_cfg["batch_size"],
                       checkpointDict["config"]['include_doc_title']) as dataset:
        logging.info(f"Extracting top k answers scores")
        res = {}
        for i, (query, answers, scores, passageIds, charOffsets) in \
                tqdm(enumerate(extractor.extract(dataset,
                                                 ext_reader_cfg["top_k_answers"],
                                                 ext_reader_cfg["max_tokens_for_answer"])),
                     total=len(dataset)):
            res[i] = {
                "raw_question": query,
                "answers": answers,
                "reader_scores": scores,
                "passages": passageIds,
                "char_offsets": charOffsets
            }

        with jsonlines.open(reader_output, "w") as wF:
            for _, record in res.items():
                wF.write(record)


def extract_predictions(reader_output, outfile):
    with jsonlines.open(reader_output, mode="r") as reader:
        with jsonlines.open(outfile, mode='w') as writer:
            logging.info("Extracting answers")
            for e in reader:
                pred_answer = e['answers'][argmax(e['reader_scores'])]
                prediction = {
                    "question": e['raw_question'],
                    "prediction": pred_answer
                }
                writer.write(prediction)


@torch.no_grad()
def run_score_aggregation(outputs, aggregation_config, aggregation_outfile):
    pipeline_data, metadata = load_pipeline_data(outputs, aggregation_config)

    pipeline_data = torch.FloatTensor(pipeline_data).transpose(-1, -2)
    parameters = {k: torch.FloatTensor(v) for k, v in aggregation_config["parameters"].items()}
    aggregated_logits = F.linear(pipeline_data, **parameters).squeeze(-1).tolist()

    with jsonlines.open(outputs["reader_output"], "r") as reader_outputs, \
            jsonlines.open(aggregation_outfile, "w") as ofwriter:
        for e, q, logits in zip(reader_outputs, metadata["questions"], aggregated_logits):
            assert q == e["raw_question"]
            e["reader_scores"] = logits
            ofwriter.write(e)


def run_fusion_extractive_abstractive(extractive_output, abstractive_output, ext_abs_fusion_outfile,
                                      ext_abs_fusion_config):
    ext_abs_data_raw, metadata = load_ext_abs_score_data(extractive_output, abstractive_output)
    ext_abs_data = torch.FloatTensor(ext_abs_data_raw)
    parameters = {k: torch.FloatTensor(v) for k, v in ext_abs_fusion_config["parameters"].items()}
    ext_abs_logits = F.linear(ext_abs_data, **parameters).squeeze(-1).tolist()

    with jsonlines.open(extractive_output, "r") as reader_outputs, \
            jsonlines.open(ext_abs_fusion_outfile, "w") as ofwriter:
        for e, q, logit, proposed_answers, scores, best_span_index in zip(reader_outputs, metadata["questions"],
                                                                          ext_abs_logits, metadata['proposed_answers'],
                                                                          ext_abs_data_raw,
                                                                          metadata['ext_best_span_idx']):
            assert q == e["raw_question"]
            assert proposed_answers[0] == e["answers"][best_span_index]
            # 0 is extractive class, 1 is abstractive class
            decision = int(logit > 0)
            e["reader_scores"] = [scores[decision]]
            e["answers"] = [proposed_answers[decision]]
            if decision:  # abstractive does not contain span info
                del e["passages"]
                del e["char_offsets"]
            else:
                e["passages"] = [e["passages"][best_span_index]]
                e["char_offsets"] = [e["char_offsets"][best_span_index]]
            ofwriter.write(e)


def has_annotation(infile):
    with jsonlines.open(infile, "r") as f:
        first_example = next(f.__iter__())
    return "answer" in first_example or "single_span_answers" in first_example


def run_predictions(infile, outfile, input_cfg=None):
    global config
    if type(input_cfg) == dict:
        config = input_cfg
    else:
        if input_cfg is None or input_cfg == "":
            input_cfg = DEFAULT_CFG_FILE
        with open(input_cfg) as fhandle:
            config = json.load(fhandle)

    input_has_annotation = has_annotation(infile)
    if input_has_annotation:
        logging.info("#" * 20 + "ATTENTION" + "#" * 20)
        logging.info("Input annotation detected! The pipeline fusions will automatically be tuned, "
                     "if parameters in the configuration are missing")
        logging.info("#" * 60)

    device = "cuda:0" if config["device"] == "gpu" else config["device"]
    config["device"] = torch.device(device if device.startswith("cuda:") and torch.cuda.is_available() else "cpu")
    mkdir(config["pipeline_cache_dir"])

    outputs = dict()
    outputs["retriever_output"] = create_filename(infile, "retrieved_outputs")
    outputs["reader_output"] = create_filename(infile, "reader_outputs")
    run_retriever(infile, outputs["retriever_output"])

    ranked_output = outputs["retriever_output"]
    if is_component_active("reranker", config):
        outputs["passage_reranker_output"] = create_filename(infile, "reranked_outputs")
        run_reranker(outputs["retriever_output"], outputs["passage_reranker_output"])
        ranked_output = outputs["passage_reranker_output"]

    run_reader(ranked_output, outputs["reader_output"])
    final_output = outputs["reader_output"]

    ### Validate reader ###
    if input_has_annotation:
        logging.info("Reader result:")
        run_eval(final_output, infile, outfile)

    if "fusion" in config:
        if not config["reader"]["active"] == "extractive":
            raise ValueError("Score aggregation is supported only when extractive reader is active.")
        fusion_config = config["fusion"]
        if is_component_active('generative_reranking_only', fusion_config) or \
                is_component_active("decide_ext_abs", fusion_config) or \
                (is_component_active("aggregate_span_scores", fusion_config) and
                 "generative_reader" in fusion_config["aggregate_span_scores"]["components"]):
            final_output = outputs["answer_reranker_output"] = create_filename(infile, "gen_reranked_reader_outputs")
            run_generative_reader_reranking(ranked_output, outputs["reader_output"], outputs["answer_reranker_output"])
            ### Validate generatively-reranked ###
            if input_has_annotation:
                logging.info("Generatively reranked extractive reader result:")
                run_eval(final_output, infile, outfile)
        if is_component_active("aggregate_span_scores", fusion_config):
            aggregation_config = fusion_config["aggregate_span_scores"]
            if is_tuning_needed(aggregation_config["parameters"]):
                aggregation_config["parameters"] = tune_score_aggregation_parameters(outputs, aggregation_config,
                                                                                     gt_file=infile)
            final_output = outputs["aggregated_output"] = create_filename(infile, "aggregated")
            run_score_aggregation(outputs, aggregation_config, outputs["aggregated_output"])

            ### Validate aggregated ###
            if input_has_annotation:
                logging.info("Aggregated extractive reader result:")
                run_eval(final_output, infile, outfile)
        if is_component_active("decide_ext_abs", fusion_config):
            ext_abs_fusion_config = fusion_config["decide_ext_abs"]
            outputs["generative_reader_output"] = create_filename(infile, "generative_reader_outputs")

            run_reader(ranked_output, outputs["generative_reader_output"], reader_type="generative")

            ### Validate generative reader ###
            if input_has_annotation:
                logging.info("Generative reader result:")
                run_eval(outputs["generative_reader_output"], infile, outfile)

            if is_tuning_needed(ext_abs_fusion_config["parameters"]):
                ext_abs_fusion_config["parameters"] = \
                    tune_ext_abs_fusion_parameters(final_output,
                                                   outputs["generative_reader_output"],
                                                   gt_file=infile)
            outputs["ext_abs_output"] = create_filename(infile, "fused_w_abs")
            run_fusion_extractive_abstractive(final_output, outputs["generative_reader_output"],
                                              outputs["ext_abs_output"], ext_abs_fusion_config)
            final_output = outputs["ext_abs_output"]

    extract_predictions(final_output, outfile)
    ### Validate final prediction ###
    if input_has_annotation:
        logging.info("Final result:")
        run_eval(final_output, infile, outfile)


def run_eval(final_output, infile, outfile):
    extract_predictions(final_output, outfile)
    metrics = evaluate_predictions(references_path=infile,
                                   predictions_path=outfile,
                                   is_regex=False)
    logging.info(f"Found {metrics['missing_predictions']} missing predictions.")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['num_correct']}/{metrics['num_total']})")
