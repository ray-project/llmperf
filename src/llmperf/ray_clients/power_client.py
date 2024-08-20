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
import logging

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

@ray.remote
class PowerLLMClient(LLMClient):
    """Client for NetMind API."""

    def __init__(self):
        pass

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        return send_req(request_config)
    
def send_req(request_config: RequestConfig) -> Dict[str, Any]:
        api_token = os.environ.get("API_TOKEN")
        if not api_token:
            raise ValueError("The environment variable API_TOKEN must be set.")
        
        if request_config.model == "llama3-8B":
            url = "https://inference-api.netmind.ai/inference-api/v1/llama3-8B"
        elif request_config.model == "llama3-70B":
            url = "https://inference-api.netmind.ai/inference-api/v1/llama3-70B"
        
        headers = {
            "Authorization": api_token,
            "Content-Type": "application/json",
        }

        prompt = request_config.prompt
        prompt, prompt_len = prompt

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
                "messages": [{"role": "user", "content": prompt}],
                "max_new_tokens": sampling_params.get("max_new_tokens", 2048),
                "stop_strings": sampling_params.get("stop_strings", ""),
                "bad_strings": sampling_params.get("bad_strings", ""),
                "temperature": sampling_params.get("temperature", 0.6),
                "top_p": sampling_params.get("top_p", 0.9),
                "top_k": sampling_params.get("top_k", 50),
                "repetition_penalty": sampling_params.get("repetition_penalty", 1.2),
            }

            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
            response_code = response.status_code
            response.raise_for_status()
            start_time = time.monotonic()

            # Process the streaming response
            token_start_time = None
            for line in response.iter_lines():
                if line:
                    if not token_start_time:
                        token_start_time = time.monotonic() - start_time

                    # Decode and accumulate the generated text
                    line_decoded = line.decode('utf-8').lstrip('data: ')
                    tokens_received += len(tokenizer.encode(line_decoded))
                    generated_text += (line_decoded + ' ')
                    
                    # Calculate inter-token latency and throughput
                    current_time = time.monotonic() - start_time
                    time_to_next_token.append(current_time - token_start_time)
                    token_start_time = current_time

            total_request_time = time.monotonic() - start_time
            tokens_received //= 2
            output_throughput = tokens_received / total_request_time

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
    client = PowerLLMClient.remote()
    request_config = RequestConfig(
        prompt=("Write a 100-word article on the Benefits of Open-Source in AI research", 10),
        model="llama3-8B",
        sampling_params={
            "temperature": 0.6,
            "max_new_tokens": 2048,
            "top_k": 50,
            "top_p": 0.9,
        },
    )
    result = send_req(request_config)
    print("RESULT:", result)
