#!/bin/bash

export HF_HOME="/efs/home/liuy72/hf-cache"

#export CUDA_VISIBLE_DEVICES=6

# Start server for Model A on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-coder-7b-instruct-v1.5 --api-key token-0 --host 0.0.0.0 --port 8001 > server1.log 2>&1 & 

# Start server for Model B on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --api-key token-1 --host 0.0.0.0 --port 8002 > server2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --api-key token-2 --host 0.0.0.0 --port 8003 > server3.log 2>&1 &
