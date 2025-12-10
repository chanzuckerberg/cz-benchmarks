"""
Example usage of KkTestTask for rare cell type detection evaluation.

This example demonstrates how to:
1. Load or create a dataset with cell type annotations
2. Generate or load cell embeddings (from a model)
3. Create task input configuration
4. Run the KkTestTask to evaluate rare cell detection
5. Interpret the results
"""

import logging
import sys
import json
import numpy as np
import anndata as ad
import pandas as pd

from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.tasks.single_cell import KkTestTask
from czbenchmarks.tasks.single_cell.kk_test import KkTestTaskInput


def create_example_data(n_cells: int = 500, n_features: int = 100):
    """Create synthetic data for demonstration.

    Creates a dataset with:
    - Common cell types (>5% frequency)
    - Rare cell types (<5% frequency)
    - Very rare cell types (<1% frequency)

    Args:
        n_cells: Number of cells to generate
        n_features: Number of embedding features

    Returns:
        AnnData object with embeddings and cell type labels
    """
    # Generate random embeddings (in practice, these would come from your model)
    X = np.random.randn(n_cells, n_features)

    # Create cell type labels with varying frequencies
    cell_types = []
    # Common types (>10%)
    cell_types.extend(["T_cell"] * int(n_cells * 0.40))  # 40%
    cell_types.extend(["B_cell"] * int(n_cells * 0.30))  # 30%
    cell_types.extend(["Monocyte"] * int(n_cells * 0.15))  # 15%

    # Rare types (1-5%)
    cell_types.extend(["NK_cell"] * int(n_cells * 0.04))  # 4%
    cell_types.extend(["Dendritic_cell"] * int(n_cells * 0.03))  # 3%
    cell_types.extend(["Mast_cell"] * int(n_cells * 0.02))  # 2%

    # Very rare types (<1%)
    cell_types.extend(["Stem_cell"] * int(n_cells * 0.01))  # 1%

    # Fill remaining with common type
    remaining = n_cells - len(cell_types)
    if remaining > 0:
        cell_types.extend(["T_cell"] * remaining)

    # Shuffle to avoid ordering bias
    np.random.shuffle(cell_types)

    # Create AnnData object
    obs = pd.DataFrame({"cell_type": cell_types[:n_cells]})
    adata = ad.AnnData(X=X, obs=obs)

    return adata


def main():
    """Run the example."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("KkTestTask Example: Rare Cell Type Detection")
    logger.info("=" * 80)

    # Step 1: Create or load data
    logger.info("\n[Step 1] Creating example dataset...")
    adata = create_example_data(n_cells=500, n_features=100)
    logger.info(f"Created dataset with {adata.n_obs} cells and {adata.n_vars} features")

    # Show cell type distribution
    logger.info("\nCell type distribution:")
    cell_type_counts = adata.obs["cell_type"].value_counts()
    cell_type_freqs = cell_type_counts / len(adata)
    for ct, count in cell_type_counts.items():
        logger.info(f"  {ct}: {count} cells ({cell_type_freqs[ct]*100:.2f}%)")

    # Step 2: Initialize the task
    logger.info("\n[Step 2] Initializing KkTestTask...")
    task = KkTestTask(random_seed=RANDOM_SEED)

    # Step 3: Create task input
    logger.info("\n[Step 3] Creating task input configuration...")
    task_input = KkTestTaskInput(
        obs="cell_type",  # Column name in adata.obs containing cell types
        rarity_threshold=0.05,  # Cell types with ≤5% frequency are considered rare
        min_cells=10,  # Minimum number of cells required for a type to be evaluated
        n_splits=5,  # Number of cross-validation folds
    )
    logger.info(f"Configuration:")
    logger.info(f"  - Rarity threshold: {task_input.rarity_threshold*100:.1f}%")
    logger.info(f"  - Minimum cells: {task_input.min_cells}")
    logger.info(f"  - CV folds: {task_input.n_splits}")

    # Step 4: Run the task
    logger.info("\n[Step 4] Running rare cell detection evaluation...")
    results = task.run(
        cell_representation=adata,
        task_input=task_input,
    )

    # Step 5: Interpret results
    logger.info("\n[Step 5] Results:")
    logger.info("=" * 80)

    # Group results by classifier
    results_by_classifier = {}
    for result in results:
        classifier = result.params.get("classifier", "unknown")
        if classifier not in results_by_classifier:
            results_by_classifier[classifier] = []
        results_by_classifier[classifier].append(result)

    # Print results for each classifier
    for classifier, clf_results in sorted(results_by_classifier.items()):
        logger.info(f"\nClassifier: {classifier}")
        logger.info("-" * 40)
        for result in clf_results:
            metric_name = result.metric_type.value
            value = result.value
            logger.info(f"  {metric_name}: {value:.4f}")

    # Convert to structured format for easy export
    results_dict = {
        "task": "KkTestTask",
        "configuration": {
            "rarity_threshold": task_input.rarity_threshold,
            "min_cells": task_input.min_cells,
            "n_splits": task_input.n_splits,
        },
        "results": [result.model_dump() for result in results],
    }

    # Print as JSON
    logger.info("\n" + "=" * 80)
    logger.info("Full results as JSON:")
    logger.info("=" * 80)
    print(json.dumps(results_dict, indent=2, default=str))

    # Interpretation guide
    logger.info("\n" + "=" * 80)
    logger.info("How to interpret these results:")
    logger.info("=" * 80)
    logger.info("""
Key metrics for rare cell detection:

1. F1 Score (MEAN_FOLD_F1_SCORE):
   - Harmonic mean of precision and recall
   - Range: 0 (worst) to 1 (perfect)
   - Good for imbalanced datasets like rare cell detection

2. Balanced Accuracy (MEAN_FOLD_BALANCED_ACCURACY):
   - Average of recall on rare and common cells
   - Range: 0 (worst) to 1 (perfect)
   - Accounts for class imbalance

3. MCC - Matthews Correlation Coefficient:
   - Considers true/false positives and negatives
   - Range: -1 (worst) to 1 (perfect), 0 is random
   - Particularly good for imbalanced datasets

4. Precision (MEAN_FOLD_PRECISION):
   - Of predicted rare cells, how many are actually rare?
   - Range: 0 (worst) to 1 (perfect)

5. Recall (MEAN_FOLD_RECALL):
   - Of actual rare cells, how many did we detect?
   - Range: 0 (worst) to 1 (perfect)

Classifier recommendations:
- Logistic regression: Fast, interpretable baseline
- KNN: Good for capturing local structure
- Random forest: Often best performance, handles non-linearity

Higher scores indicate better ability to detect rare cell types!
    """)


if __name__ == "__main__":
    main()
