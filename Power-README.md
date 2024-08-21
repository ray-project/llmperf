
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
--model  "meta-llama/Llama-3-70b-chat-hf" \
--max-num-completed-requests 3 \
--timeout 6000 \
--num-concurrent-requests 3 \
--results-dir "result_outputs" \
--llm-api "togetherai" \
--additional-sampling-params '{}'
```


### Triton
```bash
python3 token_benchmark_ray.py \
--model  "llama3" \
--max-num-completed-requests 1 \
--timeout 6000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api "localhost" \
--additional-sampling-params '{"max_tokens": 1024, "max_new_tokens": 1024}'
```