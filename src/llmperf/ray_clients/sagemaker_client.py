import io
import json
import os
import time
from typing import Any, Dict

import boto3
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class SageMakerClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            print(
                f"No AWS_ACCESS_KEY_ID found in the environment. Use the default AWS credentials."
            )
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            print(
                f"No AWS_SECRET_ACCESS_KEY found in the environment. Use the default AWS credentials."
            )
        region = os.environ.get("AWS_REGION", None)
        if not region:
            print(
                f"No AWS_REGION found in the environment. Use the default AWS credentials."
            )

        is_messages_api = os.environ.get("MESSAGES_API", "false").lower() == "true"
        is_jumpstart = os.environ.get("JUMPSTART", "false").lower() == "true"

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        if is_jumpstart or is_messages_api:
            message = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ]
        else:
            message = prompt

        model = request_config.model
        sm_runtime = boto3.client("sagemaker-runtime", region_name=region)

        sampling_params = request_config.sampling_params

        if "max_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]

        payload = {
            "inputs": message,
            "parameters": {
                **request_config.sampling_params,
            },
            "stream": True,
        }

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            response = sm_runtime.invoke_endpoint_with_response_stream(
                EndpointName=model,
                ContentType="application/json",
                Body=json.dumps(payload),
                CustomAttributes="accept_eula=true" if is_jumpstart else "",
            )

            event_stream = response["Body"]
            json_byte = b""
            generated_text = prompt
            start_json = b"{"

            for line, ttft, _ in LineIterator(event_stream):
                time_to_next_token.append(
                    time.monotonic() - most_recent_received_token_time
                )
                most_recent_received_token_time = time.monotonic()
                if line != b"" and start_json in line:
                    data = json.loads(line[line.find(start_json) :].decode("utf-8"))
                    generated_text += data["token"]["text"]
            ttft = ttft - start_time
            total_request_time = time.monotonic() - start_time
            if is_jumpstart:
                raise NotImplementedError("No tests for Jumpstart yet")
            tokens_received = len(self.tokenizer(generated_text).input_ids)
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            error_msg = str(e)
            error_response_code = 500

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config


class LineIterator:
    """
    A helper class for parsing the byte stream input.
    Reference: https://aws.amazon.com/blogs/machine-learning/elevating-the-generative-ai-experience-introducing-streaming-support-in-amazon-sagemaker-hosting/
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0
        self.ttft = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                if self.ttft == 0:
                    self.ttft = time.monotonic()
                self.read_pos += len(line)
                return line[:-1], self.ttft, time.monotonic()
            # kyle: dealing with last ']' for chat output
            if line and self.read_pos == self.buffer.getbuffer().nbytes - 1:
                self.read_pos += 1
                return line, self.ttft, time.monotonic()
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                print("Unknown event type:" + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])
