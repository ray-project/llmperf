# Benchmark: Yi 1.5 34B Chat on NVIDIA A10G

Benchmarking the performance of LLMs on the Llama 3 8b Instruct model using the NVIDIA A10G GPU using `llmperf`. The engines tested include vLLM, Hugging Face TGI, all measueed via HTTP and their OpenAI API implementations. The tests were run on an Amazon EC2 g5.12xlarge instance equipped with an NVIDIA A10G GPU.

## Test Environment
- **Instance Type**: Amazon EC2 g5.2xlarge
- **GPU**: NVIDIA A10G
- **Setup**: Requests and containers were run on the same machine via localhost.
- **Engines Tested**: 
  - [vLLM](https://docs.vllm.ai/en/stable/)
  - [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/en/index)
- **Model**: [01-ai/Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat)
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

The benchmark tested the Yi 1.5 34B Chat model on an 4x NVIDIA A10G GPU using llmperf, comparing vLLM, Hugging Face TGI, and NVIDIA NIM on an Amazon EC2 g5.12xlarge instance. Metrics included throughput, first time to token, and inter-token latency under varying levels of concurrency.


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



### Concurrency User 16



### Concurrency User 64



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
    --model 01-ai/Yi-1.5-34B-Chat \
    --tensor-parallel-size 4
```

2. Run the benchmark:

```bash
# pwd
# >/home/ubuntu/llmperf
python scripts/benchmark.py --model-id "01-ai/Yi-1.5-34B-Chat" 
```

### Hugging Face TGI

1. Start the TGI Container:

```bash
docker run --gpus all -ti -p 8000:80 \
  -e MODEL_ID="01-ai/Yi-1.5-34B-Chat" \
  -v ~/.cache/huggingface/hub:/data \
  --shm-size 1g \
  -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \
  -e MAX_INPUT_LENGTH=4000 \
  -e NUM_SHARD=4 \
  -e MAX_TOTAL_TOKENS=4096 \
  -e MAX_BATCH_PREFILL_TOKENS=8192 \
  ghcr.io/huggingface/text-generation-inference:latest
```

1. Run the benchmark:

```bash
# pwd
# >/home/ubuntu/llmperf
python scripts/benchmark.py --model-id "01-ai/Yi-1.5-34B-Chat"
```
