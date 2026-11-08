"""Synthetic interface demonstration, not biological validation or model scoring.

Run with the repository on PYTHONPATH. No external data or network is needed.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd

from czbenchmarks.datasets.perturbation_identity_cells import (
    aggregate_anndata_responses,
)
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.utils import load_custom_dataset


def main():
    cells = ad.AnnData(
        np.array([[2, 3, 5], [4, 2, 14], [3, 2, 5], [2, 4, 14]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "donor": ["d1", "d1", "d2", "d2"],
                "target": ["NTC", "A", "NTC", "A"],
                "context": ["synthetic"] * 4,
            },
            index=["cell0", "cell1", "cell2", "cell3"],
        ),
        var=pd.DataFrame(index=["g1", "g2", "denominator_only_gene"]),
    )
    options = dict(
        donor_key="donor",
        target_key="target",
        control_key="target",
        control_value="NTC",
        context_key="context",
        target_catalog=["A", "B"],
        scored_genes=["g1", "g2"],
        expected_normalization_genes=cells.var_names.tolist(),
        chunk_size=2,
    )
    panel = aggregate_anndata_responses(cells, **options)
    # Caller chooses eligibility. A real analysis should prespecify thresholds
    # and also inspect matched-control counts in panel.clone_counts.
    eligible = panel.available & (panel.min_clone_cell_counts >= 1)
    truth = panel.to_labeled_responses(eligible)
    print("Complete catalog coverage:", panel.units.tolist())
    print("Available rows:", panel.available.tolist())
    print("Selected response rows (not cells):", truth.units.tolist())
    print(truth.values)
    # Missing B stays unavailable, never imputed as zero-response.
    assert np.isnan(panel.responses[~panel.available]).all()

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.h5ad"
        cells.write_h5ad(path)
        dataset = load_custom_dataset(
            "identity_cells_example",
            custom_dataset_kwargs={
                "_target_": "czbenchmarks.datasets.perturbation_identity_cells.PerturbationIdentityCellsDataset",
                "path": str(path),
                "organism": Organism.HUMAN,
                "task_inputs_dir": str(Path(tmp) / "panel"),
                **options,
            },
        )
        np.testing.assert_allclose(
            dataset.panel.responses, panel.responses, equal_nan=True
        )
        dataset.store_task_inputs()

    # Supply independent matched/alternative/generic predictions and documented
    # fitting/holdout provenance to an existing response scorer separately.
    # This input bridge does not create predictions or run the official native
    # PerturbationExpressionPrediction task, which has a different estimand.


if __name__ == "__main__":
    main()
