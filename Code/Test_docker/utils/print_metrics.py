import os
import json
import argparse

def print_metrics(json_fname):
    # Open and read the JSON file
    with open(json_fname, 'r') as file:
        data = json.load(file)

    print(f'Name: {data["name"]}')

    # print case
    print("\n cases:")
    for case in data['cases']:
        print(case)

    # print aggregated results
    print("\n aggregated results:")
    aggregated_results = data['aggregates']
    for k, v in aggregated_results.items():
        print(f"\t{k: <{20}}: {v['mean']:.5f} +- {v['std']:.5f} | 30%: {v['30']:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='L2R Evaluation script\n'
                                                 'Docker PATHS:\n'
                                                 'DEFAULT_INPUT_PATH = Path("/input/")\n'
                                                 'DEFAULT_GROUND_TRUTH_PATH = Path("/opt/evaluation/ground-truth/")\n'
                                                 'DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path("/output/metrics.json")')
    parser.add_argument("-i", "--input", dest="input_json",
                        help="path to metrics json file")
    args = parser.parse_args()
    print_metrics(args.input_json)
