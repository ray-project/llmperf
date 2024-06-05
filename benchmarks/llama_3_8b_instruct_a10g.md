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
  - Requests Per Second: Evaluates the number of requests that can be handled by the model per second.

The benchmarking was performed using `llmperf`, a tool designed to evaluate the performance of LLMs across different frameworks and hardware configurations.

## Benchmark Results

### Concurrency User 1 

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 137.2919661 | 138.9137787 | 135.4107646 |
| Througput (token/sec)          | 31.92462559 | 32.78526142 | 32.2123514  |
| Inter Token Latency (ms/token) | 30.65149844 | 29.86407376 | 30.3319248  |


### Concurrency User 4

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 171.3956358 | 212.6501531 | 173.6120437 |
| Througput (token/sec)          | 110.9478713 | 110.7551778 | 115.3847403 |
| Inter Token Latency (ms/token) | 33.88657168 | 33.60044702 | 31.56057292 |

### Concurrency User 16

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 302.7480913 | 475.7047288 | 336.220663  |
| Througput (token/sec)          | 289.873427  | 277.873219  | 298.7441355 |
| Inter Token Latency (ms/token) | 42.66842311 | 42.95979633 | 38.68509632 |

### Concurrency User 64

| Engine                         | vLLM        | TGI         | NVIDIA NIM  |
| ------------------------------ | ----------- | ----------- | ----------- |
| First Time To Token (ms)       | 1080.420167 | 2371.579404 | 1814.533666 |
| Througput (token/sec)          | 301.1851391 | 304.3837829 | 310.8465793 |
| Inter Token Latency (ms/token) | 61.72701229 | 60.59072025 | 52.95298819 |


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