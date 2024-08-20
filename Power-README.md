
### Power

```bash
export API_TOKEN="xxx"

python3 token_benchmark_ray.py \
--model  "llama3-8B" \
--max-num-completed-requests 3 \
--timeout 6000 \
--num-concurrent-requests 3 \
--results-dir "result_outputs" \
--llm-api "power" \
--additional-sampling-params '{}'

```

### Together AI

```bash
export TOGETHERAI_API_KEY="xxx"

python3 token_benchmark_ray.py \
--model  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" \
--max-num-completed-requests 3 \
--timeout 6000 \
--num-concurrent-requests 3 \
--results-dir "result_outputs" \
--llm-api "togetherai" \
--additional-sampling-params '{}'
```