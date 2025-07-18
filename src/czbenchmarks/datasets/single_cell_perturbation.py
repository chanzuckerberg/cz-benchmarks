from typing import Dict, Optional
import io
from pathlib import Path
import numpy as np
import pandas as pd
from .single_cell import SingleCellDataset
from .types import Organism


class SingleCellPerturbationDataset(SingleCellDataset):
    """
    Single cell dataset with perturbation data, containing control and
    perturbed cells.

    This class extends `SingleCellDataset` to handle datasets with perturbation
    data. It includes functionality for validating split values, condition formats,
    and storing perturbation truth data.

    Input data requirements:
    - H5AD file containing single cell gene expression data.
    - Must have a condition column in `adata.obs` specifying control ("ctrl") and
      perturbed conditions.
    - Must have a split column in `adata.obs` to identify test samples.
    - Condition format must be one of:
      - ``ctrl`` for control samples.
      - ``{gene}+ctrl`` for single gene perturbations.
      - ``{gene1}+{gene2}`` for combinatorial perturbations.

    Attributes:
        perturbation_truth (Dict[str, pd.DataFrame]): Control data for each condition.
    """

    perturbation_truth: Dict[str, pd.DataFrame]

    def __init__(
        self,
        path: Path,
        organism: Organism,
        condition_key: str = "condition",
        split_key: str = "split",
        task_inputs_dir: Optional[Path] = None,
    ):
        """
        Initialize a SingleCellPerturbationDataset instance.

        Args:
            path (Path): Path to the dataset file.
            organism (Organism): Enum value indicating the organism.
            condition_key (str): Key for the column in `adata.obs` specifying conditions.
                Defaults to "condition".
            split_key (str): Key for the column in `adata.obs` specifying splits.
                Defaults to "split".
            task_inputs_dir (Optional[Path]): Directory for storing task-specific inputs.
        """
        super().__init__("single_cell_perturbation", path, organism, task_inputs_dir)
        self.condition_key = condition_key
        self.split_key = split_key

    def load_data(self) -> None:
        """
        Load the dataset and populate perturbation truth data.

        This method validates the presence of `condition_key` and `split_key` in
        `adata.obs`, and extracts control data for each condition into the
        `perturbation_truth` attribute.

        Raises:
            ValueError: If `condition_key` or `split_key` is not found in `adata.obs`.
        """
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
        self.adata = self.adata[self.adata.obs[self.condition_key] == "ctrl"].copy()

    def store_task_inputs(self) -> Path:
        """
        Store perturbation truth data in JSON files.

        This method saves the control data for each condition in the
        `perturbation_truth` attribute to JSON files. The filenames are dynamically
        generated based on the condition keys.

        Returns:
            Path: Path to the directory storing the task input files.
        """
        buffer = io.StringIO()
        for key, df in self.perturbation_truth.items():
            buffer = io.StringIO()
            df.to_json(buffer)
            self._store_task_input(f"perturbation_truths/{key}.json", buffer.getvalue())
        return self.task_inputs_dir

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates the following:
        - Split values must be one of {"train", "test", "val"}.
        - Condition format must be one of:
          - ``ctrl`` for control samples.
          - ``{gene}+ctrl`` for single gene perturbations.
          - ``{gene1}+{gene2}`` for combinatorial perturbations.

        Raises:
            ValueError: If invalid split values or condition formats are found.
        """
        super()._validate()

        # Validate split values
        valid_splits = {"train", "test", "val"}
        splits = set(self.adata.obs[self.split_key])
        invalid_splits = splits - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid split value(s): {invalid_splits}")

        # Validate condition format
        conditions = (
            set(self.adata.obs[self.condition_key]) | self.perturbation_truth.keys()
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
