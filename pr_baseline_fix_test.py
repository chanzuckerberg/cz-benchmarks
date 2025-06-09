from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import run_inference
from czbenchmarks.tasks import PerturbationTask
from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets.types import DataType
import pandas as pd
import numpy as np

# === Config
RAW_DATASET_NAME = "norman_perturb"
MODEL_NAME = "SCGENEPT"
MODEL_KEY = "norman"
MODEL_VARIANT = "scgenept_ncbi+uniprot_gpt"
PERTURBATION = "POU3F2+ctrl"

# === Load and hydrate dataset
print("🧠 Loading and re-hydrating dataset...")
dataset = load_dataset(RAW_DATASET_NAME)
dataset = run_inference(
    MODEL_NAME,
    dataset=dataset,
    gene_pert=PERTURBATION,
    dataset_name=MODEL_KEY,
    model_variant=MODEL_VARIANT,
    use_gears=True,
)

# === Test baseline types
try:
    adata = dataset.get_input(DataType.ANNDATA)
except KeyError:
    raise RuntimeError("❌ Could not find ANNDATA in dataset. Aborting.")

def run_and_eval(baseline_type: str):
    print(f"\n🔬 Running baseline_type='{baseline_type}'")
    task = PerturbationTask()
    try:
        task.set_baseline(dataset, gene_pert=PERTURBATION, baseline_type=baseline_type)
        results = task.run(dataset)
        print(f"✅ Completed baseline_type='{baseline_type}'")
    except Exception as e:
        print(f"❌ Failed baseline_type='{baseline_type}': {e}")
        return

    for r in results["BASELINE"]:
        print(f"{r.metric_type.value:25s} | subset: {r.params['subset']:7s} | value: {r.value:.5f}")

# === Run all 3 baselines with metrics ===
for btype in ["median", "mean", "non_control_mean"]:
    run_and_eval(btype)
