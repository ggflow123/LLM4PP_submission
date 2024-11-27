
![logo](./llm4pp_logo.png)

# LLM4PP @ PPoPP 2025

Welcome to the PPoPP Contest on LLM-based Parallel Programming (LLM4PP @ PPoPP 2025)!

* [Introduction](#introduction)
* [Objective](#objective)
* [Problem Definition](#problem-definition)
* [Starting Toolkit](#starting-toolkit)
* [Submission Guidelines](#submission-guidelines)
* [Scoring](#scoring)
* [References](#references)
* [Contact](#contact)
  
<!--Starting Toolkit for the LLM4PP competition, modified from the starting toolkit from [LLM4HWDesign](https://nvlabs.github.io/LLM4HWDesign/problem.html) is [here](https://github.com/GATECH-EIC/LLM4HWDesign_Starting_Toolkit).-->


## Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in generating high-quality content from natural language prompts, sparking growing interest in their application to parallel programming. 

## Objective

LLM4PP'25 challenges participants to develop cost-efficient techniques for using LLMs to generate fast and correct parallel programs for shared-memory multicore processors. Solutions will be evaluated based on their ability to improve performance on *code optimization* tasks where your solution will be given a correct, but potentially inefficient, sequential program and must generate an equivalent optimized parallel program. 

Competitors are invited to employ diverse methods, including:

1. Use of novel prompting strategies and workflows.
2. Development of effective strategies for fine-tuning parallel code generation models.
3. Collection or generation of new code samples for training / fine-tuning.

## Problem Definition


<!--You may employ diverse methods to accomplish this task including:
* Use of closed-source LLMs such as ones from OpenAI, and Microsoft via API access.
* Use open-source LLMs such as CodeLlama or Deepseek-Coder using local inference via vLLM.

**Restrictions on closed-source models.** Methods that employ closed-source LLMs must be *cost-efficient* and capable of completing the evaluation datasets without exceeding a strict cost-budget. For example, solutions should be able to complete the *ParEval* dataset whilst spending less than $2 in OpenAI/Gemini compute credits. Precise cost limits will be announced prior to the release of additional evaluation datasets. 

**Restrictions on open-source models.** Open-source models must be capable of running on (high-end) consumer-grade hardware for inference. Models must be capable of performing inference on a machine with: 128GB DRAM, 512GB NVME storage, and a modern Nvidia GPU with 24GB vRAM.-->

Participants may submit solutions to one or more of LLM4PP's two problem categories, which are divided based on whether solutions use proprietary closed-source models:

* **Problem 1: Dollar-constrained workflow design**. Develop a *cost-efficient* prompting workflow using LLM APIs, such as OpenAI API, to teach LLMs to solve parallel programming problems. Solutions will consist of a workflow script that will be evaluated on a set of parallel programming problems from our in-house benchmarks. Solutions must respect a strict cost limit that will be set for each evaluation dataset, and provided at the time of the dataset's release/announcement. In general, these cost limits will be set to reasonable values that allow for creative solutions (i.e., at least 10x higher than the cost of a naive solution).

* **Problem 2: Fine-tuning of memory-constrained models**. Develop techniques for prompting/fine-tuning existing open-source models, such as CodeLlama and Deepseek Coder, that improve their ability to optimize parallel programs. Solutions will consist of a workflow script (as in Problem 1) for evaluating the model along with any accompanying artifacts needed to run the submitted code such as fine-tuned model weights, knowledge database, etc. A suggested strategy is to finetune an open-source LLM on a dataset that the participants collect or generate. We will include a sample fine-tuning script, a suggested data example format, as well as sample inference/evaluation scripts in the starting toolkit. To make the comparison fair, the inference/evaluation will be constrained to a GPU memory budget of approximately 24GB of vRAM (specific hardware used for evaluation will be announced at a later date).

Solutions for both problems should provide workflow scripts that perform inference using the LLM4PP driver code, which facilitates the evaluation of solutions using diverse datasets/benchmarks. An example workflow script is provided below for illustration:

**Example workflow script**
```python
driver = LLM4PPDriver()

for problem in driver:
    problem : LLM4PP_Problem

    # TODO(participant): You must implement "logic_for_optimizing_code(problem.source_code)"
    optimized_code = logic_for_optimizing_code(problem.source_code)

    submission = LLM4PP_Submission(problem=problem,
                                   submitted_code=optimized_code)
    try:
        response = driver.submit(submission)
    except Exception as e:
        print(f"skipping problem due to exception: {e}")
driver.evaluate()
```

<!--The specific task for LLM4PP'25 is *code optimization for multicores* where the model is given
a source code file in C++ and is asked to optimize the code to improve its
performance on a shared-memory multicore machine. 

Your solution must interface with the LLM4PP evaluation scripts which requires
you to develop a solution that can accept *unoptimized* C++ source code and
transform it into *optimized* C++ source code that remains correct.-->


## Starting Toolkit

This repository contains a **starting tookit** that will help you develop and
evaluate your solutions.  The repository is organized into the two
subdirectories `model_eval/` and `model_finetune/` which contain code
for evaluating your model and fine-tuning local models
respectively.

The `model_eval/` directory contains code for evaluating your developed solution using the LLM4PP drivers. An *example dataset* based on [ParEval](https://github.com/parallelcodefoundry/ParEval) is provided to help you test your solutions during development. There are
three example *workflow scripts* provided that illustrate how to interface different types of solutions with the LLM4PP driver: 

* `evaluation.py`: a generic script with placeholder logic that simply copies the original source code.
* `evaluation_openai.py`: a solution for *Problem 1* that uses the OpenAI API to perform inference with *gpt-4o-mini* to generate optimized source code; and, 
* `evaluation_vllm.py`: a solution for *Problem 2* that uses a fine-tuned model to perform local inference using vLLM to generate optimized source code.

Additional instructional details on configuring and running the evaluation scripts are provided in [model_eval/README.md](model_eval/README.md).

The `model_finetune/` directory contains starting code for fine-tuning an open-source LLM. Additional instructional details are provided in [model_finetune/README.md](model_finetune/README.md). 

<!--**Example evaluation datasets.** The starting toolkit presently includes a single example evaluation dataset based on the [ParEval](https://github.com/parallelcodefoundry/ParEval) benchmark. Additional evaluation datasets will be made available during the competition, and will be accessed/evaluated using the LLM4PP driver interface.-->

<!--**Example `evaluation.py`**
```python
driver = LLM4PPDriver()

for problem in driver:
    problem : LLM4PP_Problem

    # TODO(participant): You must implement "logic_for_optimizing_code(problem.source_code)"
    optimized_code = logic_for_optimizing_code(problem.source_code)

    submission = LLM4PP_Submission(problem=problem,
                                   submitted_code=optimized_code)
    try:
        response = driver.submit(submission)
    except Exception as e:
        print(f"skipping problem due to exception: {e}")
driver.evaluate()
```-->

## Submission Guidelines
For each problem, submit a python file similar to `model_eval/evaluation.py` which is described in more detail [README](model_eval/README.md). If using open-source models, please also upload them to HuggingFace and make them public.

For the first problem of using closed-source LLMs, please make sure that the costs incurred by running the script are reasonable. Specific cost limits will be announced upon the release of additional datasets.

For the second problem of using open-source LLMs, please make sure that the code can be run on a single mid-to-high tier consumer-grade GPU (e.g., a GPU with approximately 24GB of vRAM).

## Scoring

Your submissions will be evaluated by running a workflow script that uses the provided LLM4PP driver code to submit optimized code for problems in our in-house Speedcode benchmark suite. The provided starter kit includes compatible LLM4PP driver code that evaluates your submissions on the [ParEval](https://github.com/parallelcodefoundry/ParEval) benchmark. This example dataset will be useful for testing your solutions.

Your submission's score will be computed, for each program $R$, using the formula $f(R)=max(1, T_{ref} / T_{sub})$ where $T_{ref}$ is the runtime of the original reference code and $T_{sub}$ is the runtime of the code submitted by your model. If your submitted solution produces incorrect code for problem R, then $f(R)=1$. This scoring methodology is designed to model the case in which a developer seeks to optimize a program that has reliable correctness tests that allows them to discard inefficient or incorrect solutions suggested by a model. When performing evaluation on a collection of N programs, your submission's score will be the geometric mean of $f(R_i)$ for $i=0,...,N-1$.


## References
1. [Gpt-4 technical report](https://arxiv.org/abs/2303.08774)
2. [HPC-Coder](https://arxiv.org/html/2306.17281v2): Modeling Parallel Programs using Large Language Models, ISC 2024
3. [Can Large Language Models Write Parallel Code?](https://arxiv.org/pdf/2401.12554.pdf) [[ParEval](https://github.com/parallelcodefoundry/ParEval)], HPDC 2024
4. [Learning Performance Improving Code-Edits](https://arxiv.org/abs/2302.07867). [[PIE](https://pie4perf.com/)], ICLR 2024

## Contact
* Xuhao Chen, cxh@mit.edu 
* Ryan Deng, ryandeng@mit.edu
* Tim Kaler, tfk@mit.edu


