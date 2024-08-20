import json
import os
import time
from typing import Any, Dict

import ray
import requests
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from together import Together


tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

@ray.remote
class TogetherAIClient(LLMClient):
    """Client for Together AI API."""

    def __init__(self):
        # Initialize the tokenizer for approximating token count
        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        print("DEBUG: TogetherAIClient initialized")

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        return send_request(request_config)

def send_request(request_config: RequestConfig) -> Dict[str, Any]:
    api_token = os.environ.get("TOGETHERAI_API_KEY")
    if not api_token:
        raise ValueError("The environment variable TOGETHER_API_KEY must be set.")
    
    # Define the URL for the request
    url = "https://api.together.xyz/v1/chat/completions"
    
    # Define the headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    prompt = request_config.prompt
    prompt, prompt_len = prompt
    model = request_config.model
    time_to_next_token = []
    tokens_received = 0
    ttft = 0
    generated_text = ""
    output_throughput = 0
    total_request_time = 0
    metrics = {}
    metrics[common_metrics.ERROR_CODE] = None
    metrics[common_metrics.ERROR_MSG] = ""

    try:
        # Define the data payload
        sampling_params = request_config.sampling_params
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": sampling_params.get("max_new_tokens", 2048),
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 0.7),
            "top_k": sampling_params.get("top_k", 50),
            "repetition_penalty": sampling_params.get("repetition_penalty", 1.0),
            "stop": sampling_params.get("stop", ["<|eot_id|>","<|eom_id|>"]),
            "stream": sampling_params.get("stream", False),
            "stream_tokens": sampling_params.get("stream", False),
        }  

        # Make the POST request
        start_time = time.monotonic()
        response = requests.post(url, headers=headers, data=json.dumps(data))
        total_request_time = time.monotonic() - start_time
        response_code = response.status_code
        response.raise_for_status()

        # Extract the generated text and tokenize it
        response_json = response.json()

        # Extract the generated text
        generated_text = response_json['choices'][0]['message']['content']

        print(generated_text)

        print("DEBUG: input data", data)
        print("DEBUG: generated_text", generated_text)

        tokens_received = len(tokenizer.encode(generated_text))
        print("DEBUG: tokens_received", tokens_received)
        print("DEBUG: total_request_time", total_request_time)
        ttft = -1  # Time to first token; adjust this if your endpoint provides this info
        output_throughput = tokens_received / total_request_time
        time_to_next_token = [
            total_request_time / tokens_received for _ in range(tokens_received)
        ]

    except Exception as e:
        metrics[common_metrics.ERROR_MSG] = str(e)
        metrics[common_metrics.ERROR_CODE] = response_code if 'response_code' in locals() else None
        print(f"Warning Or Error: {e}")

    metrics[common_metrics.INTER_TOKEN_LAT] = time_to_next_token
    metrics[common_metrics.TTFT] = ttft
    metrics[common_metrics.E2E_LAT] = total_request_time
    metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
    metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
    metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
    metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

    return metrics, generated_text, request_config

if __name__ == "__main__":
    request_config = RequestConfig(
        prompt=("Write a 100-word article on the Benefits of Open-Source in AI research", 10),
        model="meta-llama/Llama-3-8b-chat-hf",
        sampling_params={
            "temperature": 0.7,
            "max_new_tokens": 2480,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.0,
            "stop": ["<|eot_id|>","<|eom_id|>"]
        },
    )
    result = send_request(request_config)
    print("RESULT:", result)
