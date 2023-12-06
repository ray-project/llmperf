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
class VertexAIClient(LLMClient):
    """Client for VertexAI API."""

    def __init__(self):
        # VertexAI doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        project_id = os.environ.get("GCLOUD_PROJECT_ID")
        region = os.environ.get("GCLOUD_REGION")
        endpoint_id = os.environ.get("VERTEXAI_ENDPOINT_ID")
        access_token = os.environ.get("GCLOUD_ACCESS_TOKEN").strip()
        if not project_id:
            raise ValueError("the environment variable GCLOUD_PROJECT_ID must be set.")
        if not region:
            raise ValueError("the environment variable GCLOUD_REGION must be set.")
        if not endpoint_id:
            raise ValueError(
                "the environment variable VERTEXAI_ENDPOINT_ID must be set."
            )
        if not access_token:
            raise ValueError(
                "the environment variable GCLOUD_ACCESS_TOKEN must be set."
            )
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
            # Define the URL for the request
            url = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"
            )

            # Define the headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            sampling_params = request_config.sampling_params
            if "max_new_tokens" in sampling_params:
                sampling_params["maxOutputTokens"] = sampling_params.pop(
                    "max_new_tokens"
                )

            # Define the data payload
            data = {"instances": [{"prompt": prompt}], "parameters": sampling_params}

            # Make the POST request
            start_time = time.monotonic()
            response = requests.post(url, headers=headers, data=json.dumps(data))
            total_request_time = time.monotonic() - start_time
            response_code = response.status_code
            response.raise_for_status()
            # output from the endpoint is in the form:
            # {"predictions": ["Input: ... \nOutput:\n ..."]}
            generated_text = response.json()["predictions"][0].split("\nOutput:\n")[1]
            tokens_received = len(self.tokenizer.encode(generated_text))
            ttft = -1
            output_throughput = tokens_received / total_request_time
            time_to_next_token = [
                total_request_time / tokens_received for _ in range(tokens_received)
            ]

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            metrics[common_metrics.ERROR_CODE] = response_code
            print(f"Warning Or Error: {e}")
            print(response_code)
            print(response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = time_to_next_token
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


if __name__ == "__main__":
    # Run these before hand:

    # gcloud auth application-default login
    # gcloud config set project YOUR_PROJECT_ID
    # export GCLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)
    # export GCLOUD_PROJECT_ID=YOUR_PROJECT_ID
    # export GCLOUD_REGION=YOUR_REGION
    # export VERTEXAI_ENDPOINT_ID=YOUR_ENDPOINT_ID

    client = VertexAIClient.remote()
    request_config = RequestConfig(
        prompt=("Give me ten interview questions for the role of program manager.", 10),
        model="gpt3",
        sampling_params={
            "temperature": 0.2,
            "max_new_tokens": 256,
            "top_k": 40,
            "top_p": 0.95,
        },
    )
    ray.get(client.llm_request.remote(request_config))
