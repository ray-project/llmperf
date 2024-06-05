# Benchmark: Llama 3 8b Instruct on NVIDIA A10G

Benchmarking the performance of LLMs on the Llama 3 8b Instruct model using the NVIDIA A10G GPU using `llmperf`. The engines tested include vLLM, Hugging Face TGI, and NVIDIA NIM, all measueed via HTTP and their OpenAI API implementations. The tests were run on an Amazon EC2 g5.2xlarge instance equipped with an NVIDIA A10G GPU.

## Test Environment
- **Instance Type**: Amazon EC2 g5.2xlarge
- **GPU**: NVIDIA A10G
- **Setup**: Requests and containers were run on the same machine via localhost.
- **Engines Tested**: 
  - [vLLM](https://docs.vllm.ai/en/stable/)
  - [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/en/index)
  - [NVIDIA NIM](https://build.nvidia.com/)
- **Model**: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- **Scenario**:
  - Expected Input: 550 tokens (mean)
  - Expected Output: 150 tokens (mean)
  - Concurrent Requests: 2, 4, 8, 16, 32, 64
- **metrics**: 
  - Throughput: Measures how many tokens can be processed in a given time frame.
  - First Time to Token: Tracks the time taken to generate the first token in response to a request.
  - Latency (Inter-Token Latency): Measures the time elapsed between generating successive tokens.

The benchmarking was performed using `llmperf`, a tool designed to evaluate the performance of LLMs across different frameworks and hardware configurations.

## Benchmark Results

The benchmark tested the Llama 3 8b Instruct model on an NVIDIA A10G GPU using llmperf, comparing vLLM, Hugging Face TGI, and NVIDIA NIM on an Amazon EC2 g5.2xlarge instance. Metrics included throughput, first time to token, and inter-token latency under varying levels of concurrency.

NVIDIA NIM offers overall best engine, #1 in inter-token latency and maintaining competitive throughput. For example, at concurrency 64, NIM had the lowest inter-token latency (52.95 ms/token) and the highest throughput (310.85 tokens/sec). vLLM showed the fastest first token generation across all tests, such as 137.29 ms at concurrency 1, making it ideal for latency-sensitive applications. 

Hugging Face TGI maintained high throughput, like 304.38 tokens/sec at concurrency 64, making it suitable for high-load scenarios. However, TGI often lagged behind in first token generation and inter-token latency. For instance, at concurrency 16, TGI’s first token generation time was 475.70 ms, 57.13% slower than vLLM (302.75 ms) and 41.54% slower than NIM (336.22 ms).

TGI's performance metrics evolved noticeably. At 1 concurrent user, TGI had a first time to token of 138.91 ms and throughput of 32.79 tokens/sec. At 64 users, TGI's first time to token ( 2371.58 ms) is dramatically higher compared to vLLM (1080.42 ms) and NVIDIA NIM (1814.53 ms)

Despite this increase, TGI managed to keep a high throughput (304.38 tokens/sec) compared to vLLM's 301.19 tokens/sec and NVIDIA NIM's 310.85 tokens/sec. TGI's inter-token latency is also stays competitive, though it is still outperformed by NVIDIA NIM, especially at 64 users where TGI had 60.59 ms/token compared to NVIDIA NIM's 52.95 ms/token.
As concurrency increased from 1 to 64 users, TGI's inter-token latency remains close but slightly better to vLLM, being 2.57% faster at 1 user and 1.84% faster at 64 users.



### Concurrency User 1 

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 137.2919661 | 138.9137787 | 135.4107646 |
| Througput (token/sec)          | 31.92462559 | 32.78526142 | 32.2123514  |
| Inter Token Latency (ms/token) | 30.65149844 | 29.86407376 | 30.3319248  |

- For First Time To Token (ms), TGI is 1.18% slower than vLLM and NVIDIA NIM is 1.37% faster than vLLM. Compared to TGI, NVIDIA NIM is 2.52% faster.- 
- For Throughput (token/sec), TGI is 2.70% slower than vLLM and NVIDIA NIM is 0.90% slower than vLLM. Compared to TGI, NVIDIA NIM is 1.75% faster.
- For Inter Token Latency (ms/token), TGI is 2.57% faster than vLLM and NVIDIA NIM is 1.04% faster than vLLM. Compared to TGI, NVIDIA NIM is 1.57% slower.


### Concurrency User 4

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 171.3956358 | 212.6501531 | 173.6120437 |
| Througput (token/sec)          | 110.9478713 | 110.7551778 | 115.3847403 |
| Inter Token Latency (ms/token) | 33.88657168 | 33.60044702 | 31.56057292 |

- For First Time To Token (ms), TGI is 24.07% slower than vLLM and NVIDIA NIM is 1.29% slower than vLLM. Compared to TGI, NVIDIA NIM is 18.36% faster.
- For Throughput (token/sec), TGI is 0.17% faster than vLLM and NVIDIA NIM is 4.00% slower than vLLM. Compared to TGI, NVIDIA NIM is 4.18% slower.
- For Inter Token Latency (ms/token), TGI is 0.84% faster than vLLM and NVIDIA NIM is 6.86% faster than vLLM. Compared to TGI, NVIDIA NIM is 6.07% faster.


### Concurrency User 16

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 302.7480913 | 475.7047288 | 336.220663  |
| Througput (token/sec)          | 289.873427  | 277.873219  | 298.7441355 |
| Inter Token Latency (ms/token) | 42.66842311 | 42.95979633 | 38.68509632 |

- For First Time To Token (ms), TGI is 57.13% slower than vLLM and NVIDIA NIM is 11.06% slower than vLLM. Compared to TGI, NVIDIA NIM is 29.32% faster.
- For Throughput (token/sec), TGI is 4.14% faster than vLLM and NVIDIA NIM is 3.06% slower than vLLM. Compared to TGI, NVIDIA NIM is 7.51% slower.
- For Inter Token Latency (ms/token), TGI is 0.68% slower than vLLM and NVIDIA NIM is 9.34% faster than vLLM. Compared to TGI, NVIDIA NIM is 9.95% faster.


### Concurrency User 64

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 1080.420167 | 2371.579404 | 1814.533666 |
| Througput (token/sec)          | 301.1851391 | 304.3837829 | 310.8465793 |
| Inter Token Latency (ms/token) | 61.72701229 | 60.59072025 | 52.95298819 |

- For First Time To Token (ms), TGI is 119.51% slower than vLLM and NVIDIA NIM is 67.95% slower than vLLM. Compared to TGI, NVIDIA NIM is 23.49% faster.
- For Throughput (token/sec), TGI is 1.06% slower than vLLM and NVIDIA NIM is 3.21% slower than vLLM. Compared to TGI, NVIDIA NIM is 2.12% slower.
- For Inter Token Latency (ms/token), TGI is 1.84% faster than vLLM and NVIDIA NIM is 14.21% faster than vLLM. Compared to TGI, NVIDIA NIM is 12.61% faster.



## Steps to Run Each Benchmark

Make sure to login into huggingface to have access to Llama 3 8B Instruct model with `huggingface-cli login`. We are going to use the [benchmark.py](../scripts/benchmark.py) script to run the benchmarks. The script will run the benchmark for 2, 4, 8, 16, 32, 64, and 128 concurrent requests using the same configuration for each engine.

### vLLM 

1. Start the vLLM Container:
```bash
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e "HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.4.3 \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

2. Run the benchmark:

```bash
# pwd
# >/home/ubuntu/llmperf
python scripts/benchmark.py --model-id "meta-llama/Meta-Llama-3-8B-Instruct" 
```

### Hugging Face TGI

1. Start the TGI Container:

```bash
docker run --gpus all -ti -p 8000:80 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
  -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \
  -e MAX_INPUT_LENGTH=6000 \
  -e MAX_TOTAL_TOKENS=6144 \
  -e MAX_BATCH_PREFILL_TOKENS=8192 \
  ghcr.io/huggingface/text-generation-inference:2.0.4
```

1. Run the benchmark:

```bash
# pwd
# >/home/ubuntu/llmperf
python scripts/benchmark.py --model-id "meta-llama/Meta-Llama-3-8B-Instruct"
```

### NVIDIA NIM (llm_engine: tensorrt_llm)

NIM Config:
```bash
Profile metadata: feat_lora: false
Profile metadata: precision: fp16
Profile metadata: tp: 1
Profile metadata: llm_engine: tensorrt_llm
Profile metadata: pp: 1
Profile metadata: profile: throughput
Profile metadata: gpu: A10G
```
_Note: NVIDIA NIM requires a valid license and nv api key. Make sure to replace `NGC_API_KEY`. 

1. Start the NVIDIA NIM Container:

```bash
docker run --gpus all -ti -p 8000:8000 \
  -e NGC_API_KEY=nvapi-xxxx  \
  nvcr.io/nim/meta/llama3-8b-instruct:1.0.0
```

1. Run the benchmark:
_Note: NVIDIA changed the name from the official model id_

```bash
# pwd
# >/home/ubuntu/llmperf
python scripts/benchmark.py --model-id "meta/llama3-8b-instruct"
```