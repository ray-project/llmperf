import argparse
import glob
import json

# python parse_results.py --results-dir "result_outputs"

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", help="Directory containing the result files")
args = parser.parse_args()

# Check if --results-dir argument is provided
if not args.results_dir:
  print("Please provide the --results-dir argument.")
  exit(1)

# Reads the summary.json file and prints the results
with open(glob.glob(f'{args.results_dir}/*summary.json')[0], 'r') as file:
  data = json.load(file)

print(f"Avg. Input token length: {data['mean_input_tokens']}")
print(f"Avg. Output token length: {data['mean_output_tokens']}")
print(f"Avg. First-Time-To-Token: {data['results_ttft_s_mean']*1000:.2f}ms")
print(f"Avg. Thorughput: {data['results_mean_output_throughput_token_per_s']:.2f} tokens/sec")
print(f"Avg. Latency: {data['results_inter_token_latency_s_mean']*1000:.2f}ms/token")