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


@ray.remote
class TritonClient(LLMClient):
    """Client for VertexAI API."""

    def __init__(self):
        # Triton doesn't return the number of tokens that are generated in offline modeso we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        address = os.environ.get("TRITON_HOST").strip()
        access_token = os.environ.get("ACCESS_TOKEN").strip()
        if not address:
            raise ValueError(
                "the environment variable TRITON_HOST must be set."
            )
        if not access_token:
            raise ValueError(
                "the environment variable ACCESS_TOKEN must be set."
            )

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        # Define the data payload
        body = {
            "text_input": prompt,
            "stream": False,
            "bad_words": "",
            "stop_words": "",
            "max_tokens": 2056,
        }

        sampling_params = request_config.sampling_params
        for k, v in sampling_params.items():
            body[k]=v
        
        # Define the URL for the request
        if body['stream']:
            url = f"http://{address}/v2/models/{request_config.model}/generate_stream"
        else:
            url = f"http://{address}/v2/models/{request_config.model}/generate"
        # Define the headers
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Make the POST request
        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            if body['stream']:
                with requests.post(
                    url,
                    json=body,
                    stream=True,
                    timeout=180,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_msg = response.text
                        error_response_code = response.status_code
                        response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=None):
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        stem = "data: "
                        chunk = chunk[len(stem) :]
                        if chunk == b"[DONE]":
                            continue
                        tokens_received += 1
                        data = json.loads(chunk)

                        if "error" in data:
                            error_msg = data["error"]["message"]
                            error_response_code = data["error"]["code"]
                            raise RuntimeError(data["error"]["message"])
                            
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += data["text_output"]

                total_request_time = time.monotonic() - start_time
                output_throughput = tokens_received / total_request_time
            
            else:
                # Make the POST request
                response = requests.post(url, headers=headers, data=json.dumps(body))
                total_request_time = time.monotonic() - start_time
                response_code = response.status_code
                response.raise_for_status()
                # output from the endpoint is in the form:
                # data: {"model_name":"","model_version":"","sequence_end":,"sequence_id":,"sequence_start":,"text_output":""}
                generated_text = response.json()["text_output"]
                tokens_received = len(self.tokenizer.encode(generated_text))
                ttft = -1

                output_throughput = tokens_received / total_request_time
                time_to_next_token = [
                    total_request_time / tokens_received for _ in range(tokens_received)
                ]
        
        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


if __name__ == "__main__":
    # Run these before hand:
    
    # export ADDRESS=YOUR_HOST_ADDRESS
    # export ACCESS_TOKEN=YOUR_ENDPOINT_ID


    client = TritonClient.remote()
    request_config = RequestConfig(
        prompt=("Give me ten interview questions for the role of program manager.", 11),
        model="ensemble",
        sampling_params={
            "stream": True,
            # "temperature": 0.2,
            "max_tokens": 1,
            # "top_k": 40,
            # "top_p": 0.95,
        },
    )
    output = ray.get(client.llm_request.remote(request_config))
    print(output)
