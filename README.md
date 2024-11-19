# LLM4PP @ PPoPP 2025

Welcome to the PPoPP Contest on LLM-based Parallel Programming (LLM4PP @ PPoPP 2025)!

* [Introduction](#introduction)
* [Objective](#objective)
* [Problem Definition](#problem-definition)
* [Scoring](#scoring)
* [Starting Toolkit](#starting-toolkit)
* [Submission Guidelines](#submission-guidelines)
* [References](#references)
* [Contact](#contact)
  
<!--Starting Toolkit for the LLM4PP competition, modified from the starting toolkit from [LLM4HWDesign](https://nvlabs.github.io/LLM4HWDesign/problem.html) is [here](https://github.com/GATECH-EIC/LLM4HWDesign_Starting_Toolkit).-->

## Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in generating high-quality content from natural language prompts, sparking growing interest in their application to parallel programming. 

<!--
Despite the significant potential and community excitement, current state-of-the-art pretrained LLMs, such as OpenAI's GPT-4 [1], still struggle to produce practical parallel code without extensive human intervention in their original forms. 
In parallel code generation, for example, these models tend to either generate non-compilable or non-functional code, necessitating human correction, or produce overly simplistic or inefficient parallel implementations. 
This issue can primarily stem from the LLMs' limited exposure to parallel code data during pretraining. A pioneering attempt in HPC-Coder [2] demonstrates that using a large-scale parallel code dataset can improve LLMs' PP abilities. 
However, models trained with publicly available datasets are still far behind human experts. Thus, developing open-source, high-quality, PP-specific code datasets is essential for unlocking the full potential of LLM-based PP.
This year's contest seeks to address this challenge by asking you to help build a large-scale, high-quality parallel code generation dataset. 
By open-sourcing this dataset, we aim to establish critical infrastructure for advancing LLM-based parallel programming workflows. 
Participants will be invited to author a short paper summarizing our efforts, insights, and lessons learned, thereby paving the way for future initiatives.

Unfortunately, the development of LLM for PP is severely hindered by the scarcity of high-quality, publicly accessible parallel code datasets. 
Specifically, the lack of adequate datasets prevents effective fine-tuning of LLMs, a critical method for equipping them with PP domain knowledge and mitigating their limited exposure to PP-specific data during pretraining. 
This shortage thus significantly impedes progress in LLM-based parallel code generation.
-->

## Objective
The goal of this contest is to 
(1) explore methods (e.g., prompting and finetuning) that leverages LLMs to solve parallel programming problems.
(2) collect or generate parallel code samples with data cleaning techniques, to facilitate the development of more effective LLM-based parallel code generation through fine-tuning. 
<!--enrich the current parallel code dataset to a large-scale, high-quality open-source dataset, facilitating the development of more effective LLM-based parallel programming through fine-tuning. 
Participants are asked to (1) collect or generate parallel code samples and (2) enhance the dataset quality through data cleaning and label generation techniques. 
Participants' contributions will be evaluated based on the improvement their data brings to the fine-tuned LLM.-->

## Problem Definition

## Scoring

### Base Dataset
The base dataset used in the contest is our [LLM4PP dataset](https://huggingface.co/datasets/speedcode/LLM4PP_dataset) from [Leetcode problems](https://leetcode.com/problemset/). 
For your submitted data, please follow the same format as the [LLM4PP dataset](https://huggingface.co/datasets/speedcode/LLM4PP_dataset). 

## Starting Toolkit

To get started, participants are provided the starting toolkit, which is this github repository.
It includes (1) an existing dataset as the base dataset, 
(2) an example dataset of parallel code from external sources providing the format example of participants' submission, 
(3) a codebase to fine-tune a specific LLM with the base dataset and the example submission dataset, 
(4) an evaluation script to measure the how the example submission dataset mitigate the bias of the base dataset.

<!--
and (5) the deduplication codebase we will use to duplicate the repeated data samples. 
Participants are expected to just replace the example submission dataset with their own collected datasets and get the corresponding metric from the starting toolkit to further improve their datasets during Phase I.
-->

### Toolkit Release Progress
<!-- - [x] **Deduplication**: Scripts to identify and remove duplicate samples from the dataset. -->
- [x] **Fine-tuning**: Scripts to fine-tune a pretrained language model on the base dataset.
- [x] **Evaluation**: Tools to evaluate the performance of the fine-tuned model using standard metrics.


### Setup Environment

We assume CUDA 12.1. (Only needed if you want to do fine-tuning and evaluation on your own.)

`conda env create -f environment.yml`

<!--
## Deduplication
The toolkit includes a deduplication script, which will be used to deduplicate each participant's data against the base dataset during the evaluation of Phase I.
To run the deduplication script:
```bash
python minhash.py
```
-->

### Evaluation

The following shows an example on how to evaluate your fine-tuned model using ParEval.

**Prerequisites**:

`export HF_TOKEN=your_huggingface_token`

Prepare your fine-tuned model and tokenizer in HuggingFace format.

```bash
pip install -e pareval
```

**Evaluation Scripts**:

```bash
cd model_eval
python evaluation.py <path_to_folder_with_your_model_and_config> <your_huggingface_token>
#example: python evaluation.py "finetuned_model/" "hf-xxxxxxxxxx"
```

NOTE: The folder with your model and config should include two files (1) the generated pytorch_model.bin and 
(2) the [model config](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base/blob/main/config.json) of Deepseek Coder 6.7B from HuggingFace.

The results will be printed and logged in `./model_eval/data/gen.jsonl`

## Submission Guidelines

## References
1. [Gpt-4 technical report](https://arxiv.org/abs/2303.08774)
2. [HPC-Coder](https://arxiv.org/html/2306.17281v2): Modeling Parallel Programs using Large Language Models, ISC 2024
3. [Can Large Language Models Write Parallel Code?](https://arxiv.org/pdf/2401.12554.pdf) [[ParEval](https://github.com/parallelcodefoundry/ParEval)], HPDC 2024
4. [Learning Performance Improving Code-Edits](https://arxiv.org/abs/2302.07867). [[PIE](https://pie4perf.com/)], ICLR 2024

## Contact
* Xuhao Chen, cxh@mit.edu 
* Ryan Deng, ryandeng@mit.edu 
