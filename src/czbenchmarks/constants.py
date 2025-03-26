import pathlib

# Base paths
INPUT_DATA_PATH_DOCKER = "/input/data.dill"
RAW_INPUT_DIR_PATH_DOCKER = "/raw"
OUTPUT_DATA_PATH_DOCKER = "/output/data.dill"
ARTIFACTS_PATH_DOCKER = "/artifacts"
DATASETS_CACHE_PATH = "~/.cz-benchmarks/datasets"
MODEL_WEIGHTS_PATH_DOCKER = "/weights"
MODEL_WEIGHTS_CACHE_PATH = "~/.cz-benchmarks/weights"
CZ_BENCHMARKS_MODELS_ECR_REGISTRY = "339713142298.dkr.ecr.us-west-2.amazonaws.com"

# Derived constants
def get_numbered_path(base_path: str, index: int) -> str:
    """
    Get numbered version of a path
    (e.g., /path/to/data.dill-> /path/to/data_1.dill)
    """
    path = pathlib.Path(base_path)
    stem = path.stem  # 'data'
    suffix = path.suffix  # '.dill'
    return str(path.parent / f"{stem}_{index}{suffix}")


def get_base_name(path: str) -> str:
    """Get the base filename pattern (e.g., /path/to/data.dill -> data*.dill)"""
    path = pathlib.Path(path)
    return f"{path.stem}*{path.suffix}"
