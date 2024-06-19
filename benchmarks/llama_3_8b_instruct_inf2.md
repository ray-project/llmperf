# Benchmark: Llama 3 8b Instruct on AWS Inferentia2

Benchmarking the performance of LLMs on the Llama 3 8b Instruct model using the AWS Inferentia2 using `llmperf`.

## Test Environment
- **Instance Type**: Amazon EC2 inf2.xlarge
- **GPU**: AWS Inferentia2
- **Setup**: Requests and containers were run on the same machine via localhost.
- **Engines Tested**: 
  - [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/en/index)
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

The first time to token (ms) at the 50th percentile increases significantly from 1001 ms for one user to 7503 ms for 16 users, indicating higher latency with more users. 

Throughput (tokens per second at the 50th percentile) improves with concurrency, peaking at 142.72 tokens per second for 16 users. 

Inter-token latency (ms/token) at the 50th percentile rises from 52.18 ms for one user to 97.17 ms for 16 users, showing longer intervals between tokens as user count increases.


| Concurrent Users               | 1     | 2     | 4      | 8     | 16     |
|--------------------------------|-------|-------|--------|-------|--------|
| Mean Input Token               | 550   | 550   | 550    | 550   | 550    |
| Mean Output Token              | 177   | 175   | 174    | 176   | 175    |
| First Time To Token (ms) p50   | 1001  | 1419  | 3732.78| 7539  | 7503   |
| Throughput (token/sec) p50     | 16.23 | 29.2  | 48.08  | 72.16 | 142.72 |
| Inter Token Latency (ms/token) p50 | 52.18 | 58.29 | 69.93  | 93.95 | 97.17  |
| Request per minute             | 6.4   | 11.45 | 18.81  | 27.78 | 26.9   |
| Errors                         | 0     | 0     | 0      | 0     | 56     |
| Cost per 1M token              | $13.01| $7.23 | $4.39  | $2.93 | $1.48  |



## Steps to Run Each Benchmark

Make sure to login into huggingface to have access to Llama 3 8B Instruct model with `huggingface-cli login`. We are going to use the [benchmark.py](../scripts/benchmark.py) script to run the benchmarks. The script will run the benchmark for 2, 4, 8, 16, 32, 64, and 128 concurrent requests using the same configuration for each engine.


1. Start the TGI Container:

```bash
docker run --privileged -ti -p 8000:80 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  -e HF_AUTO_CAST_TYPE="fp16" \
  -e HF_NUM_CORES=2 \
  -e MAX_BATCH_SIZE=8 \
  -e MAX_INPUT_LENGTH=4000 \
  -e MAX_TOTAL_TOKENS=4096 \
  ghcr.io/huggingface/neuronx-tgi:0.0.23
```

1. Run the benchmark:

```bash
# pwd
# >/home/ubuntu/llmperf
python scripts/benchmark.py --model-id "meta-llama/Meta-Llama-3-8B-Instruct"
```
