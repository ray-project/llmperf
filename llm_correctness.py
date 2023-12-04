import argparse
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import num2words
import ray
from tqdm import tqdm

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    LLMPerfResults,
)

MAX_RANDOM_NUMBER = 10000


def llm_correctness(
    model: str,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="chat",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The type of request to make. Either "chat" or "litellm".

    Returns:
        A tuple containing summary metrics and raw results from the test.

    """

    if not additional_sampling_params:
        additional_sampling_params = {}

    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)
    start_time = time.monotonic()

    num_errored_requests = 0
    num_mismatched_requests = 0
    num_completed_requests = 0

    sampling_params = {"temperature": 0.0}
    sampling_params.update(additional_sampling_params)
    completed_requests = []
    iter = 0
    pbar = tqdm(total=max_num_completed_requests)
    while (
        time.monotonic() - start_time < test_timeout_s
        and num_completed_requests < max_num_completed_requests
    ):
        iter += 1
        rnd_number = random.randint(0, MAX_RANDOM_NUMBER)
        rnd_num_words = num2words.num2words(rnd_number)

        prompt = f"Convert the following sequence of words into a number: {rnd_num_words}.\nPrint the number first."

        request_config = RequestConfig(
            model=model,
            prompt=(prompt, 0),
            sampling_params=sampling_params,
            metadata={"rnd_number": rnd_number},
            llm_api=llm_api,
        )
        req_launcher.launch_requests(request_config)

        if not (iter % num_concurrent_requests):
            completed_requests.extend(req_launcher.get_next_ready())
        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    raw_results = []

    print("Mismatched and errored requests.")
    for out in completed_requests:
        metrics, generated_text, completed_request_config = out

        raw_results.append(
            {
                "metrics": metrics,
                "generated_text": generated_text,
                "request_config": dict(completed_request_config),
            }
        )

        # if there were no errors when making request.
        if not metrics[common_metrics.ERROR_CODE]:
            try:
                commas_between_numbers_re = r"(\d+),(?=\d)"
                gen_text_commas_removed = re.sub(
                    commas_between_numbers_re, r"\1", generated_text
                )
                nums = re.findall(r"\d+", gen_text_commas_removed)
                generated_text = gen_text_commas_removed.replace("\n", " ")

                assert str(completed_request_config.metadata["rnd_number"]) in nums
            except:
                num_mismatched_requests += 1
                print(
                    f"    mismatched request: {generated_text}, expected: {completed_request_config.metadata['rnd_number']}"
                )
        else:
            num_errored_requests += 1
            print(
                f"    The request errored: {metrics[common_metrics.ERROR_CODE]}, "
                f"{metrics[common_metrics.ERROR_MSG]} "
            )
    print()

    error_rate = num_errored_requests / num_completed_requests
    mismatch_rate = num_mismatched_requests / num_completed_requests
    num_non_errored_requests = num_completed_requests - num_errored_requests
    summary_metrics = {}
    summary_metrics[common_metrics.NUM_ERRORS] = num_errored_requests
    summary_metrics["num_mismatched_requests"] = num_mismatched_requests
    summary_metrics["error_rate"] = error_rate
    summary_metrics["mismatch_rate"] = mismatch_rate
    summary_metrics[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    summary_metrics["num_non_errored_requests"] = num_non_errored_requests

    # Metadata
    summary_metrics["model"] = model
    summary_metrics["num_concurrent_requests"] = num_concurrent_requests
    summary_metrics["additional_sampling_params"] = additional_sampling_params
    summary_metrics["llm_api"] = llm_api

    return summary_metrics, raw_results


def run(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, str],
):
    """
    Args:
        llm_api: The type of request to make. Either "chat" or "litellm".
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.

    """

    summary_metrics, raw_results = llm_correctness(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
    )

    time.sleep(2)

    print(
        f"Results for llm correctness test for {model} queried with the {llm_api} api."
    )
    print(
        f"Errors: {summary_metrics[common_metrics.NUM_ERRORS]}, "
        f"Error rate: {summary_metrics['error_rate']}"
    )

    print(
        f"Mismatched: {summary_metrics['num_mismatched_requests']}, "
        f"Mismatch rate: {summary_metrics['mismatch_rate']}"
    )
    print(f"Completed: {summary_metrics[common_metrics.NUM_COMPLETED_REQUESTS]}")
    print(f"Completed without errors: {summary_metrics['num_non_errored_requests']}")

    if results_dir:
        file_name = f"{model}_correctness"
        file_name = re.sub(r"[^\w\d-]+", "-", file_name)
        file_name = re.sub(r"-{2,}", "-", file_name)
        summary_file_name = f"{file_name}_summary"
        individual_responses_filename = f"{file_name}_individual_responses"
        summary_metrics.update(user_metadata)
        results = LLMPerfResults(name=summary_file_name, metadata=summary_metrics)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")
        with open(results_dir / f"{summary_file_name}.json", "w") as f:
            json.dump(results.to_dict(), f, indent=4)
        with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
            json.dump(raw_results, f, indent=4)


args = argparse.ArgumentParser(description="Run a correctness test for a given model.")

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send. (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=50,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The type of request to make. The supported llm apis are {SUPPORTED_APIS} "
        " (`default: %(default)s`)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

if __name__ == "__main__":
    args = args.parse_args()

    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
    )
