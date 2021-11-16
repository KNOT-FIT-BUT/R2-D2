# Introduction

This repository contains the official implementation accompanying our
- **EMNLP'21 Findings paper** [R2-D2: A Modular Baseline for Open-Domain Question Answering](https://aclanthology.org/2021.findings-emnlp.73/)
- **preprint** [Pruning the Index Contents for Memory Efficient Open-Domain QA](https://arxiv.org/abs/2102.10697).  


The sources present in this repository can be used to __run model inference in the pipeline__. Therefore the model uses already pretrained checkpoints.
Please note our work is accompanied with two repositories. If you are interested in __training new models__ instead, check the [scalingQA](https://github.com/KNOT-FIT-BUT/scalingQA) repository.

If you use **R2-D2**, please cite our paper:
```
@inproceedings{fajcik-etal-2021-r2-d2,
    title = "{R2-D2}: {A} {M}odular {B}aseline for {O}pen-{D}omain {Q}uestion {A}nswering",
    author = "Fajcik, Martin  and
      Docekal, Martin  and
      Ondrej, Karel  and
      Smrz, Pavel",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.73",
    pages = "854--870",
    abstract = "This work presents a novel four-stage open-domain QA pipeline R2-D2 (Rank twice, reaD twice). The pipeline is composed of a retriever, passage reranker, extractive reader, generative reader and a mechanism that aggregates the final prediction from all system{'}s components. We demonstrate its strength across three open-domain QA datasets: NaturalQuestions, TriviaQA and EfficientQA, surpassing state-of-the-art on the first two. Our analysis demonstrates that: (i) combining extractive and generative reader yields absolute improvements up to 5 exact match and it is at least twice as effective as the posterior averaging ensemble of the same models with different parameters, (ii) the extractive reader with fewer parameters can match the performance of the generative reader on extractive QA datasets.",
}
```

If you use our **corpus pruning approach** from our pre-print, please cite our preprint:

```
@article{fajcik2021pruning,
  title={{P}runing the {I}ndex {C}ontents for {M}emory {E}fficient {O}pen-{D}omain {QA}},
  author={Fajcik, Martin and Docekal, Martin and Ondrej, Karel and Smrz, Pavel},
  journal={arXiv preprint arXiv:2102.10697},
  year={2021}
}
```

## Table of Contents
- [Installing and configuring the environment](#installing-and-configuring-the-environment)
- [Running the pipeline](#running-the-pipeline)
- [What is in the configuration](#what-is-in-the-configuration)
- [Tuning the component fusion](#tuning-the-component-fusion)



## Installing and configuring the environment
Set your system's locale.

```shell
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

Install required packages (including scalingQA) via  __python3.6__'s pip using `requirements.txt`.

```shell
python -m pip install -r requirements.txt
```


## Running the pipeline
The pipeline contains shell scripts in the following format:
```
[dev|test]_[NQopen|TRIVIAopen|efficientQA].sh <configuration_file>
```
You can use these scripts to replicate the results presented in our paper. We include various configuration files in `configurations/pipeline/[dataset]`.
To replicate the results of our _full_ and _pruned_ pipeline, you might likely be interested in configurations `r2d2_full.json` and `r2d2_pruned.json`.

For example:  
* Run `test_NQopen.sh configurations/pipeline/NQ/r2d2_pruned.json` to download Natural-Questions Open test data and evaluate the results.   
* Run `dev_TRIVIAopen.sh configurations/pipeline/Trivia/r2d2_full.json` to download TriviaQA-Open development data and get the results.

**!** Please note that when using the full configuration, the whole embedding matrix is loaded into RAM (64.6GB for full 21M passage index).


## What is in the configuration
Here we explain what are the contents of the configuration file using `configurations/pipeline/NQ/r2d2_full.json` as an example.
```python
{
  # Options of the pipeline's index (therefore the embeddings and the database).
  "index": {
    "passage_embeddings": {
       # Directory, where embeddings are located in, w.r.t. current PYTHONPATH.
      "directory": ".index", 
       # Zipped embedding matrix in h5 format. Matrix may be saved in fp16, but is casted fo fp32 after loading.
      "name": "DPR_nqsingle_official.h5.zip",
       # An url where to download the file. May be left as empty string, if the directory is prepared explicitly and already contains the file.
      "url": "r2d2.fit.vutbr.cz/index/nq-open/DPR_nqsingle_official.h5.zip"
    },
    "database": {
      # Directory, where database is located, w.r.t. current PYTHONPATH.
      "directory": ".index",
      # Zipped SQLite database.
      "name": "wiki2018_dpr_blocks.db.zip",
      # URL to SQLite database. May be left as empty string, if explicitly prepared in the directory.
      "url": "r2d2.fit.vutbr.cz/data/wiki2018_dpr_blocks.db.zip"
    }
  },
  "retriever": {
    # Directory, where checkpoint is located, w.r.t. current PYTHONPATH.
    "directory": ".checkpoints",
    # Zipped checkpoint name. Checkpoint may be saved in fp16, and is automatically casted to fp32 on loading.
    "name": "dpr_official_questionencoder_nq.pt.zip",
    # URL to checkpoint's remote location. May be left empty as in previous cases.
    "url": "r2d2.fit.vutbr.cz/checkpoints/nq-open/dpr_official_questionencoder_nq.pt.zip",
    # How many top-k passage indices to output.
    "top_k": 200
  },
  "reranker": {
    # Where the reranker is active. If set to "false" the reranking step will be skipped.
    "active": "true",
    "directory": ".checkpoints",
    "name": "reranker_roberta-base_2021-02-25-17-27_athena19_HIT@25_0.8299645997487725.ckpt.zip",
    "url": "r2d2.fit.vutbr.cz/checkpoints/nq-open/reranker_roberta-base_2021-02-25-17-27_athena19_HIT@25_0.8299645997487725.ckpt.zip",
    "config": {
      # minibatch size used during the inference
      "batch_size": 100
    }
  },
  "reader": {
    # what reader is active, in case of using both readers (RXD2 configurations) case, keep "extractive" as active.
    "active": "extractive",
    "extractive": {
      "directory": ".checkpoints",
      "name": "EXT_READER_ELECTRA_LARGE_B128_049_J24.pt.zip",
      "url": "r2d2.fit.vutbr.cz/checkpoints/nq-open/EXT_READER_ELECTRA_LARGE_B128_049_J24.pt.zip",
      "config": {
        # minibatch size used during the inference
        "batch_size": 24,
        # how many top-k answers to return per question
        "top_k_answers": 50,
        # how many whitespace tokens (based on space --- s.split(" ")) can the answer have
        "max_tokens_for_answer": 5
      }
    },
    "generative": {
      "directory": ".checkpoints",
      "name": "generative_reader_EM0.4830_S7000_Mt5-large_2021-01-15_05.pt.zip",
      "url": "r2d2.fit.vutbr.cz/checkpoints/nq-open/generative_reader_EM0.4830_S7000_Mt5-large_2021-01-15_05.pt.zip"
    }
  },
  # options for component fusion
  "fusion": {
    # turn this on, if you want to use generative reader only for generative reranking of spans from extractive reader
    "generative_reranking_only": {
      "active": "false"
    },
    # score aggregation
    "aggregate_span_scores": {
      # whether it is active
      "active": "true", 
      # the parameters of the fusion
      # see section "Tuning the component fusion" below to find how to tune these parameters automatically on dev data.
      # the order of parameters is: [extractive reader, retriever, reranker, generative reader]
      "parameters": {"weight": [[0.09513413906097412, 0.2909115254878998, 0.22145800292491913, 0.7677204608917236]], "bias": [-0.25344452261924744]},
      # what components to use in fusion
      "components": [
        "retriever",
        "reranker",
        "extractive_reader",
        "generative_reader"
      ]
    },
    # binary decision
    "decide_ext_abs": {
      # whether it is active
      "active": "true",
      # the parameters of the fusion
      # see section "Tuning the component fusion" below to find how to tune these parameters automatically on dev data.
      # the order of parameters is: [extractive reader, generative reader]
      "parameters":  {"weight": [[-0.12410655617713928, 0.4435662627220154]], "bias": [-0.510728657245636]}
    }
  },
  # what device to use, set "gpu" if you want to use default gpu, or "cuda:n" if you want to use n-th gpu, or "cpu")
  "device": "gpu",
  # cache folder where transformers cache should be available, or will be downloaded to
  "transformers_cache": ".Transformers_cache",
  # folder where intermediate results of pipeline's components will be saved
  "pipeline_cache_dir": ".pipeline_cache_r2d2_full"
}
```
## Tuning the component fusion
In case you want to tune the pipeline parameters of individual fusions yourself, you might want to check out configurations in the `configurations/pipeline/[dataset]/parameter_tuning` folder. There, you will find the configurations just like in the previous section, except that the value of "parameters" for both fusions is set to:
```python
      ...
      "parameters": "tune_please!"
      ...
```
This indicates the pipeline to tune the parameters based on the input file provided. Note in this case, the input file needs to contain golden annotation for the parameter tuning. Therefore it needs to have jsonl format with lines like:
```
{"question": "what is the oath that new citizens take", "answer": ["United States Oath of Allegiance"]}
```
For example, running the command:
```bash
./dev_NQopen.sh configurations/pipeline/NQ/parameter_tuning/r2d2_pruned_nqval_tuner.json
```
will tune the pipeline on NQ-Open validation data, where the golden annotation is present.

In general, if parameters are tuned, the golden annotation ("answer" field) does not need to be present.
