# Instructions to run the code for LLM4PP contest

## 1. Create a conda environment
```python
conda create -n LLM4PP python=3.10
conda activate LLM4PP
```

## 2. Install packages via pip
```python
pip install -r requirements_lessons.txt
```

## 3. Host three opensource models with 3 GPUs with vLLM
### 1. (Optional, require resources for model hosting) For instance, on amazon instance, initiate a tmux session, then require an interactive node with 4 gpus.
```bash
srun --partition=queue-g6e12xlarge --exclusive --pty bash -i
```
### 2. Specify HuggingFace Cache Path Environment Variable. For instance,
```bash
export HF_HOME="/efs/home/liuy72/hf-cache"
```
### 3. In terminal, host 3 models on 3 GPUs one by one
```bash
# Start server for Model A on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-coder-7b-instruct-v1.5 --api-key token-0 --host 0.0.0.0 --port 8001 > server1.log 2>&1 & 
```
```bash
# Start server for Model B on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --api-key token-1 --host 0.0.0.0 --port 8002 > server2.log 2>&1 &
```
```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --api-key token-2 --host 0.0.0.0 --port 8003 > server3.log 2>&1 &
```

## 4. Check your hostname and Run the code
### 1. Check your hostname:
```bash
hostname
```
### 2. Run the code:
```bash
python multi_agent_lesson_server_factor_no_refutation.py logging.loggers.main_logger.level=INFO logging.loggers.class_logger.level=INFO Rounds=4 temperature=0.2 reason_temperature=0.2 benchmark=ParEval mode=openmp localhost=queue-g6e12xlarge-dy-g6e12xlarge-1
```
Here, to integrate which benchmark to use, refers to line 250 of ```multi_agent_lesson_server_factor_no_refutation.py```. ```mode=openmp``` means except c++, openmp is used for parallelization. ```localhost=queue-g6e12xlarge-dy-g6e12xlarge-1``` is the link referring to the models hosted on GPUs. Change it with your hostname.