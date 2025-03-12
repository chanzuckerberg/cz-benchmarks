import pytest
import numpy as np
import pandas as pd

from czbenchmarks.datasets.single_cell import (
    SingleCellDataset,
    PerturbationSingleCellDataset,
)
from czbenchmarks.datasets.types import Organism, DataType
from czbenchmarks.models.validators.scvi import SCVIValidator
from czbenchmarks.models.validators.scgenept import ScGenePTValidator
from czbenchmarks.models.validators.scgpt import ScGPTValidator
from czbenchmarks.models.validators.geneformer import GeneformerValidator
from czbenchmarks.models.validators.uce import UCEValidator
from tests.utils import create_dummy_anndata, DummyDataset


# For all fully implemented single‑cell validators, dataset validation passes on a valid benchmarking dataset fixture.
@pytest.mark.parametrize(
    "validator_class, obs_columns, var_columns, dataset_class",
    [
        (
            SCVIValidator,
            SCVIValidator.required_obs_keys,
            SCVIValidator.required_var_keys,
            SingleCellDataset,
        ),
        (
            ScGenePTValidator,
            ScGenePTValidator.required_obs_keys,
            ScGenePTValidator.required_var_keys,
            PerturbationSingleCellDataset,
        ),
        (
            ScGPTValidator,
            ScGPTValidator.required_obs_keys,
            ScGPTValidator.required_var_keys,
            SingleCellDataset,
        ),
        (
            GeneformerValidator,
            GeneformerValidator.required_obs_keys,
            GeneformerValidator.required_var_keys,
            SingleCellDataset,
        ),
        (
            UCEValidator,
            UCEValidator.required_obs_keys,
            UCEValidator.required_var_keys,
            SingleCellDataset,
        ),
    ],
)
def test_valid_dataset(validator_class, obs_columns, var_columns, dataset_class):
    validator = validator_class()
    ann = create_dummy_anndata(obs_columns=obs_columns, var_columns=var_columns)

    # Instantiate a valid dataset with the correct organism
    dataset = dataset_class("dummy_path", organism=Organism.HUMAN)
    dataset.set_input(DataType.ANNDATA, ann)
    dataset.set_input(DataType.METADATA, ann.obs)

    # For perturbation datasets, also set the required perturbation truth input.
    if dataset_class == PerturbationSingleCellDataset:
        dummy_truth = {
            "ctrl": pd.DataFrame(
                np.ones((ann.n_obs, ann.n_vars)), columns=ann.var_names
            )
        }
        dataset.set_input(DataType.PERTURBATION_TRUTH, dummy_truth)

    try:
        validator.validate_dataset(dataset)
    except Exception as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")


# 1. Dataset validation fails when the dataset_type is incompatible.
def test_invalid_dataset_type():
    validator = SCVIValidator()  # Expects a SingleCellDataset (exact type match)
    dummy_ds = DummyDataset("dummy_path")
    # Provide required inputs to bypass missing-input errors.
    ann = create_dummy_anndata(
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"]
    )
    dummy_ds.set_input(DataType.ANNDATA, ann)
    dummy_ds.set_input(DataType.METADATA, ann.obs)
    dummy_ds.set_input(DataType.ORGANISM, Organism.HUMAN)
    with pytest.raises(ValueError, match="Dataset type mismatch"):
        validator.validate_dataset(dummy_ds)


# 2. Dataset validation fails when required inputs are missing.
def test_missing_required_inputs():
    validator = SCVIValidator()  # Expects both ANNDATA and METADATA as inputs.
    dataset = SingleCellDataset("dummy_path", organism=Organism.HUMAN)
    ann = create_dummy_anndata(
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"]
    )
    dataset.set_input(DataType.ANNDATA, ann)
    # Intentionally do not set METADATA.
    with pytest.raises(ValueError, match="Missing required inputs"):
        validator.validate_dataset(dataset)


# 3. For a single‑cell model: Dataset validation fails when the organism is incompatible.
def test_incompatible_organism():
    validator = SCVIValidator()  # SCVIValidator supports HUMAN and MOUSE only.
    # Use an organism that is not supported.
    dataset = SingleCellDataset("dummy_path", organism=Organism.CHIMPANZEE)
    ann = create_dummy_anndata(
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"]
    )
    dataset.set_input(DataType.ANNDATA, ann)
    dataset.set_input(DataType.METADATA, ann.obs)
    with pytest.raises(ValueError, match="Dataset organism"):
        validator.validate_dataset(dataset)


# 4. For a single‑cell model: Dataset validation fails when required obs keys are missing.
def test_missing_required_obs_keys():
    validator = (
        SCVIValidator()
    )  # Requires obs keys: dataset_id, assay, suspension_type, donor_id.
    # Create AnnData missing one required obs key (e.g. "donor_id").
    ann = create_dummy_anndata(obs_columns=["dataset_id", "assay", "suspension_type"])
    dataset = SingleCellDataset("dummy_path", organism=Organism.HUMAN)
    dataset.set_input(DataType.ANNDATA, ann)
    dataset.set_input(DataType.METADATA, ann.obs)
    with pytest.raises(ValueError, match="Missing required obs keys"):
        validator.validate_dataset(dataset)


# 5. For a single‑cell model: Dataset validation fails when required var keys are missing.
def test_missing_required_var_keys():
    validator = GeneformerValidator()  # Requires var key: "feature_id".
    # Create AnnData that does not include "feature_id" (e.g. a different var column name).
    ann = create_dummy_anndata(var_columns=["some_other_feature"])
    dataset = SingleCellDataset("dummy_path", organism=Organism.HUMAN)
    dataset.set_input(DataType.ANNDATA, ann)
    dataset.set_input(DataType.METADATA, ann.obs)
    with pytest.raises(ValueError, match="Missing required var keys"):
        validator.validate_dataset(dataset)
