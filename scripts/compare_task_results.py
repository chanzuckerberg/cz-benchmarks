import argparse
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple
from decimal import Decimal, InvalidOperation


# ----------------------------
# Extracts the list of task_results from a JSON or YAML result file.
# The file may contain extra text outside of the serialized result (e.g., CLI logs).
# The file extension (.json, .yaml, .yml) is used as a hint to determine the format.
# ----------------------------
def extract_task_results(file_path: str) -> List[Dict[str, Any]]:
    ext = Path(file_path).suffix.lower()
    with open(file_path, "r") as f:
        content = f.read()

    if ext == ".json":
        # JSON results are expected to be a complete object embedded in text.
        try:
            start_index = content.index("{")
            end_index = content.rindex("}") + 1
            json_str = content[start_index:end_index]
            data = json.loads(json_str)
            return data.get("task_results", [])
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing JSON from {file_path}: {e}")

    elif ext in {".yaml", ".yml"}:
        # YAML results are expected to start from 'task_results:' onwards.
        lines = content.splitlines()
        try:
            start_index = next(
                i
                for i, line in enumerate(lines)
                if line.strip().startswith("task_results:")
            )
        except StopIteration:
            raise ValueError(f"'task_results:' not found in {file_path}")
        yaml_content = "\n".join(lines[start_index:])
        try:
            parsed = yaml.safe_load(yaml_content)
            return parsed.get("task_results", [])
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML from {file_path}: {e}")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# Used to uniquely identify each metric within a task.
def metric_key(metric: Dict[str, Any]) -> Tuple[str, frozenset]:
    return (metric["metric_type"], frozenset(metric.get("params", {}).items()))


# Used to uniquely identify a task_result based on its configuration.
def task_key(task: Dict[str, Any]) -> Tuple[str, str, str, frozenset]:
    return (
        task["task_name"],
        task["model_type"],
        task["dataset_name"],
        frozenset(task["model_args"].items()),
    )


# ----------------------------
# Builds a mapping of unique task_keys to their corresponding task_results
# from one or more ground truth files.
# The first occurrence of a task_key is preserved across all files.
# ----------------------------
def build_ground_truth_index(
    file_paths: List[str],
) -> Dict[Tuple, Tuple[Dict[str, Any], str]]:
    index = {}
    for file_path in file_paths:
        task_results = extract_task_results(file_path)
        for task in task_results:
            key = task_key(task)
            if key not in index:
                index[key] = (task, file_path)
    return index


# ----------------------------
# Performs a safe and configurable float comparison using Decimal.
# Returns True if the values are within the given tolerance.
# ----------------------------
def compare_decimals(
    val_a: Any, val_b: Any, tolerance: Decimal = Decimal("1e-8")
) -> bool:
    try:
        dec_a = Decimal(str(val_a))
        dec_b = Decimal(str(val_b))
        return abs(dec_a - dec_b) <= tolerance
    except InvalidOperation:
        return False


# ----------------------------
# Main comparison logic:
# - Extracts task_results from the evaluation file and all ground truth files.
# - Matches tasks by (task_name, model_type, dataset_name, model_args).
# - Compares metrics within each matched task.
# - Prints human-readable differences to stdout.
# ----------------------------
def compare_task_results(
    file_b: str, file_as: List[str], tolerance: str = "1e-8"
) -> None:
    index_a = build_ground_truth_index(file_as)
    results_b = extract_task_results(file_b)
    tol = Decimal(tolerance)
    eval_filename = Path(file_b).name

    for task_b in results_b:
        key = task_key(task_b)
        if key not in index_a:
            print(
                f"⚠️ Task in eval-file: {eval_filename} not found in any ground-truth file: {key}"
            )
            continue

        task_a, ground_truth_file = index_a[key]
        ground_truth_filename = Path(ground_truth_file).name

        metrics_a = {metric_key(m): m["value"] for m in task_a["metrics"]}
        metrics_b = {metric_key(m): m["value"] for m in task_b["metrics"]}

        for m_key, val_b in metrics_b.items():
            val_a = metrics_a.get(m_key)
            if val_a is None:
                print(
                    f"➕ Metric in eval-file: {eval_filename} missing in ground-truth-file: {ground_truth_filename} | {m_key} | Value: {val_b}"
                )
            elif not compare_decimals(val_a, val_b, tol):
                print(
                    f"🧮 Metric value differs (tolerance:{tol}) for {key} | {m_key} | ground-truth-file: {ground_truth_filename}={val_a}, eval-file: {eval_filename}={val_b}"
                )

        for m_key in metrics_a:
            if m_key not in metrics_b:
                print(
                    f"➖ Metric in ground-truth-file: {ground_truth_filename} missing in eval-file: {eval_filename} | {m_key} for task {key}"
                )


# ----------------------------
# Command-line interface:
# - --eval-file specifies the file to verify
# - --ground-truth-files accepts one or more reference files
# - --tolerance allows optional control over float comparison precision
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="""
Compare task_results in an evaluation result file against one or more ground truth files.

Both evaluation and ground truth files can be in JSON or YAML format and may contain
extra text like logs before or after the result block.

The script identifies tasks by their (task_name, model_type, dataset_name, model_args)
and compares all matching metrics by type and params, using a decimal-based float tolerance.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--eval-file",
        required=True,
        help="The result file to verify (e.g. FileB.yaml or FileB.json)",
    )

    # It might be the case there might be multiple ground truth files. For example:
    # 1. The ground truth for a particular run maybe too expensive to generate and thus
    #    split over multiple runs
    # 2. The comparison might want to legitimately compare against data from
    #    previous runs
    parser.add_argument(
        "--ground-truth-files",
        nargs="+",
        required=True,
        help="One or more ground truth result files (e.g. FileA1.yaml FileA2.json)",
    )
    parser.add_argument(
        "--tolerance",
        default="1e-8",
        help="Tolerance for decimal comparison (default: 1e-8)",
    )
    args = parser.parse_args()

    compare_task_results(
        args.eval_file, args.ground_truth_files, tolerance=args.tolerance
    )


if __name__ == "__main__":
    main()


# TODO: Enhance for new package.