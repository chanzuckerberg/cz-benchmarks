import io
from pathlib import Path
import numpy as np
import pandas as pd
from czbenchmarks.datasets.single_cell import SingleCellDataset
from czbenchmarks.datasets.types import Organism


class SingleCellPerturbationDataset(SingleCellDataset):
    """
    Single cell dataset with perturbation data, containing control and
    perturbed cells.

    Input data requirements:

    - H5AD file containing single cell gene expression data
    - Must have a condition column in adata.obs specifying control ("ctrl") and
      perturbed conditions.
    - Must have a split column in adata.obs to identify test samples
    - Condition format must be one of:

      - ``ctrl`` for control samples
      - ``{gene}+ctrl`` for single gene perturbations
      - ``{gene1}+{gene2}`` for combinatorial perturbations
    """
    
    perturbation_truth: dict[str, pd.DataFrame] = {}

    def __init__(
        self,
        path: Path,
        organism: Organism,
        # TODO: eliminate these keys and assume h5ad complies with "perturbation" schema
        condition_key: str = "condition",
        split_key: str = "split",
    ):
        super().__init__("single_cell_perturbation", path, organism)
        self.condition_key = condition_key
        self.split_key = split_key

    def load_data(self) -> None:
        super().load_data()

        if self.condition_key not in self.adata.obs.columns:
            raise ValueError(
                f"Condition key '{self.condition_key}' not found in adata.obs"
            )

        if self.split_key not in self.adata.obs.columns:
            raise ValueError(f"Split key '{self.split_key}' not found in adata.obs")

        # Store control data for each condition in the reference dataset
        conditions = np.array(list(self.adata.obs[self.condition_key]))

        test_conditions = set(
            self.adata.obs[self.condition_key][self.adata.obs[self.split_key] == "test"]
        )

        truth_data = {
            str(condition): pd.DataFrame(
                data=self.adata[conditions == condition].X.toarray(),
                index=self.adata[conditions == condition].obs_names,
                columns=self.adata[conditions == condition].var_names,
            )
            for condition in set(test_conditions)
        }

        self.perturbation_truth = truth_data
        # FIXME BYODATASET: as originally implemented, this overwrites adata from SingleCellDataset
        # TODO: Do we need to store model outputs as a file output? (e.g. store_model_input())
        self.adata = self.adata[self.adata.obs[self.condition_key] == "ctrl"].copy()

    def store_task_inputs(self) -> Path:
        # Save perturbation truth data to a file
        buffer = io.StringIO()
        for key, df in self.perturbation_truth.items():
            buffer = io.StringIO()
            df.to_json(buffer)
            self._store_task_input(f"perturbation_truth_{key}.json", buffer.getvalue())
        


    # FIXME VALIDATION: move to validation class?
    def _validate(self) -> None:
        super()._validate()

        # Validate split values
        valid_splits = {"train", "test", "val"}
        splits = set(self.adata.obs[self.split_key])
        invalid_splits = splits - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid split value(s): {invalid_splits}")

        # Validate condition format
        conditions = set(
            list(self.adata.obs[self.condition_key])
            + list(self.perturbation_truth.keys())
        )

        for condition in conditions:
            if condition == "ctrl":
                continue

            parts = condition.split("+")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid perturbation condition format: {condition}. "
                    "Must be 'ctrl', '{gene}+ctrl', or '{gene1}+{gene2}'"
                )
