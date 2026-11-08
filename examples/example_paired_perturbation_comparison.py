"""Synthetic arithmetic example; not evidence of biological performance.

Run from an installed checkout:
    python examples/example_paired_perturbation_comparison.py
"""

import json

import numpy as np

from czbenchmarks.metrics import metrics_registry
from czbenchmarks.metrics.types import MetricResult, MetricType


def main() -> None:
    truth = np.eye(2)
    result = MetricResult(
        metric_type=MetricType.PAIRED_MSE_REDUCTION,
        value=metrics_registry.compute(
            MetricType.PAIRED_MSE_REDUCTION,
            y_true=truth,
            y_pred=truth.copy(),
            y_reference=np.full((2, 2), 0.5),
            group_ids=["donor_a", "donor_b"],
        ),
        params={
            "data": "synthetic",
            "reference": "generic",
            "weighting": "equal_donor",
        },
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
