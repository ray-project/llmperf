import time
from typing import Any, Dict
import ray

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class LiteLLMClient(LLMClient):
    """Client for LiteLLM Completions API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        # litellm package isn't serializable, so we import it within the function
        # to maintain compatibility with ray.
        from litellm import completion, validate_environment

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        assert (
            request_config.llm_api is not None
        ), "the request config's llm_api must be set."
        if request_config.llm_api == "litellm":
            model = request_config.model
        else:
            model = request_config.llm_api + "/" + request_config.model
        validation_result = validate_environment(model)
        if validation_result["missing_keys"]:
            raise ValueError(
                f"The following environment vars weren't found but were necessary for "
                f"the model {request_config.model}: {validation_result['missing_keys']}"
            )
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

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

        try:
            start_time = time.monotonic()
            most_recent_received_token_time = time.monotonic()

            response = completion(**body)
            ttft = 0
            for tok in response:
                if tok.choices[0].delta:
                    delta = tok.choices[0].delta
                    if delta.get("content", None):
                        if ttft == 0:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        generated_text += delta["content"]
                        most_recent_received_token_time = time.monotonic()
                        tokens_received += 1

            total_request_time = time.monotonic() - start_time

            output_throughput = tokens_received / total_request_time

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
