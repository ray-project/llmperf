import argparse
from dataclasses import dataclass, field
import os
import subprocess
import json
import glob
import pandas as pd


@dataclass
class Config:
    model_id: str
    concurrency: list = field(default_factory=list)
    num_requests: int = 100  # Default value if not specified
    input_token_length: int = 500  # Default value if not specified
    output_token_length: int = 200  # Default value if not specified


def benchmark(config):
    """Run the performance script for each concurrency level."""
    results = {}
    detailed_results = {}
    # get script file path its ../token_benchmark_ray.py from the current benchmark.py
    script_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../token_benchmark_ray.py"
    )
    for concurrency in config.concurrency:
        print(f"Running test with concurrency: {concurrency}")
        os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
        os.environ["OPENAI_API_KEY"] = "none"
        output_dir = f'result_{config.model_id.replace("/","_")}_{concurrency}'
        cmd = [
            "python",
            script_file_path,
            "--model",
            config.model_id,
            "--mean-input-tokens",
            str(config.input_token_length),
            "--stddev-input-tokens",
            "0",
            "--mean-output-tokens",
            str(config.output_token_length),
            "--stddev-output-tokens",
            "0",
            "--max-num-completed-requests",
            str(config.num_requests),
            "--timeout",
            "600",
            "--num-concurrent-requests",
            str(concurrency),
            "--results-dir",
            output_dir,
            "--llm-api",
            "openai",
            "--additional-sampling-params",
            "{}",
        ]
        subprocess.run(cmd)
        with open(glob.glob(f"{output_dir}/*summary.json")[0], "r") as file:
            data = json.load(file)
        c_detailed_results = {
            "concurrency": concurrency,
            "mean_input_token_length": data["results_number_input_tokens_mean"],
            "mean_output_token_length": data["results_number_output_tokens_mean"],
            "time_to_first_token_in_ms_(ttft)_p50": data["results_ttft_s_quantiles_p50"]
            * 1000,
            "time_to_first_token_in_ms_(ttft)_p75": data["results_ttft_s_quantiles_p75"]
            * 1000,
            "time_to_first_token_in_ms_(ttft)_p95": data["results_ttft_s_quantiles_p95"]
            * 1000,
            "throughput_token_per_s_(token/sec)_p50": data[
                "results_request_output_throughput_token_per_s_quantiles_p50"
            ],
            "throughput_token_per_s_(token/sec)_p75": data[
                "results_request_output_throughput_token_per_s_quantiles_p75"
            ],
            "throughput_token_per_s_(token/sec)_p95": data[
                "results_request_output_throughput_token_per_s_quantiles_p95"
            ],
            "latency_ms_per_token_(inter_token_latency)_p50": data[
                "results_inter_token_latency_s_quantiles_p50"
            ]
            * 1000,
            "latency_ms_per_token_(inter_token_latency)_p75": data[
                "results_inter_token_latency_s_quantiles_p75"
            ]
            * 1000,
            "latency_ms_per_token_(inter_token_latency)_p95": data[
                "results_inter_token_latency_s_quantiles_p95"
            ]
            * 1000,
            "requests_per_minute_(qpm)": data["results_num_completed_requests_per_min"],
            "results_number_errors": data["results_number_errors"],
            "results_num_completed_requests": data["results_num_completed_requests"],
        }
        # append results
        results[concurrency] = data
        detailed_results[concurrency] = c_detailed_results
        with open(
            f'{config.model_id.replace("/","_")}_cur_{concurrency}.json', "w"
        ) as file:
            json.dump(detailed_results[concurrency], file, indent=2)
        # remove the output directory
        # subprocess.run(["rm", "-rf", output_dir])
    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(
        description="Manage Docker, run tests, and process results."
    )
    parser.add_argument("--model-id", type=str, help="The model ID to benchmark.")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        help="The concurrency levels to test. Add via space separated list.",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--num-requests", type=int, help="The number of requests to make.", default=100
    )
    parser.add_argument(
        "--input-token-length",
        type=int,
        help="The length of the input tokens.",
        default=550,
    )
    parser.add_argument(
        "--output-token-length",
        type=int,
        help="The length of the output tokens.",
        default=150,
    )
    args = parser.parse_args()

    # convert args to config
    config = Config(
        model_id=args.model_id,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        input_token_length=args.input_token_length,
        output_token_length=args.output_token_length,
    )
    # run the benchmark
    results, detailed_results = benchmark(config)
    # print the results in a nice markdown table using pandas
    df = pd.DataFrame(detailed_results)
    df.to_csv(f"{config.model_id.replace('/','_')}.csv")
    # write to csv
    print(df.to_markdown())


if __name__ == "__main__":
    main()

# example usage
# python scripts/benchmark.py --model-id "openai/chatgpt" --concurrency 1 2 --num-requests 100 --input-token-length 550 --output-token-length 150
