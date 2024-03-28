import json
import os
import time
from typing import Any, Dict

import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from transformers import AutoTokenizer


@ray.remote
class HuggingFaceTgiClient(LLMClient):
    """Client for Hugging Face TGI"""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        tokenizer = AutoTokenizer.from_pretrained(request_config.model)
        # try to apply chat template with system message if error retry without system message
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        # update prompt_len to match include special tokens
        prompt_len = len(tokenizer(prompt).input_ids)

        sampling_params = request_config.sampling_params

        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]

        body = {
            "inputs": prompt,
            "parameters": {
                **request_config.sampling_params,
            },
            "stream": True,
        }
        address = os.environ.get("HUGGINGFACE_API_BASE", "https://api-inference.huggingface.co")
        # Adds the model name to the address if it is not "local" or "inference endpoint"
        if address == "https://api-inference.huggingface.co":
            address = f"{address}/models/{request_config.model}"
        headers = {
            "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_TOKEN', '')}"
        }

        time_to_next_token = []
        tokens_received = 0
        ttft = None
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        try:
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                # ADAPTED FROM: https://github.com/huggingface/text-generation-inference/blob/6c4496a1a30f119cebd3afbfedd847039325dbc9/clients/python/text_generation/client.py#L767
                for byte_payload in response.iter_lines():
                    # Skip line
                    if byte_payload == b"\n":
                        continue
                    payload = byte_payload.decode("utf-8")
                    # Event data
                    if payload.startswith("data:"):
                        # Decode payload
                        tokens_received += 1
                        chunk = json.loads(payload.lstrip("data:").rstrip("/n"))

                        if chunk.get("token", None):
                            if not ttft:
                                ttft = time.monotonic() - start_time
                                time_to_next_token.append(ttft)
                            else:
                                time_to_next_token.append(
                                    time.monotonic() - most_recent_received_token_time
                                )
                            most_recent_received_token_time = time.monotonic()
                            generated_text += chunk["token"]["text"]

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(
            time_to_next_token
        )  # This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
