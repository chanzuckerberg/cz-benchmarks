import pytest
from czibench.datasets.base import BaseDataset
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import DataType, Organism
from czibench.models.base import BaseModelImplementation, BaseModelValidator
from czibench.models.single_cell import BaseSingleCellValidator
from czibench.models.validators.geneformer import GeneformerValidator
from czibench.models.validators.scgenept import ScGenePTValidator
from czibench.models.validators.scgpt import ScGPTValidator
from czibench.models.validators.scvi import SCVIValidator
from czibench.models.validators.uce import UCEValidator


@pytest.fixture
def valid_h5ad_path():
    return "tests/assets/example-small.h5ad"


@pytest.fixture
def single_cell_dataset_human(valid_h5ad_path):
    ds = SingleCellDataset(path=valid_h5ad_path, organism=Organism.HUMAN)
    return ds


@pytest.fixture
def single_cell_dataset_mouse(valid_h5ad_path):
    ds = SingleCellDataset(path=valid_h5ad_path, organism=Organism.MOUSE)
    return ds


@pytest.fixture
def loaded_dataset_human(single_cell_dataset_human):
    single_cell_dataset_human.load_data()
    return single_cell_dataset_human


@pytest.fixture
def loaded_dataset_mouse(single_cell_dataset_mouse):
    single_cell_dataset_mouse.load_data()
    return single_cell_dataset_mouse


def test_base_model_validator_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseModelValidator()


def test_base_model_implementation_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseModelImplementation()


class MockModel(BaseModelImplementation):
    dataset_type = SingleCellDataset

    def _validate_dataset(self, dataset: BaseDataset):
        pass

    @property
    def inputs(self):
        return {DataType.ANNDATA}

    @property
    def outputs(self):
        return {DataType.EMBEDDING}

    def get_model_weights_subdir(self):
        return "mock_model_weights"

    def _download_model_weights(self):
        pass

    def run_model(self):
        pass

    def parse_args(self):
        pass


def test_single_cell_validator_abstract_methods():
    with pytest.raises(TypeError):
        BaseSingleCellValidator()


@pytest.mark.parametrize(
    "validator_cls,expected_inputs,expected_outputs",
    [
        (UCEValidator, {DataType.ANNDATA}, {DataType.EMBEDDING}),
        (SCVIValidator, {DataType.ANNDATA, DataType.METADATA}, {DataType.EMBEDDING}),
        (GeneformerValidator, {DataType.ANNDATA}, {DataType.EMBEDDING}),
        (ScGenePTValidator, {DataType.ANNDATA}, {DataType.PERTURBATION}),
        (ScGPTValidator, {DataType.ANNDATA}, {DataType.EMBEDDING}),
    ],
)
def test_validator_io_sets(validator_cls, expected_inputs, expected_outputs):
    v = validator_cls()
    assert v.inputs == expected_inputs
    assert v.outputs == expected_outputs


@pytest.mark.parametrize("validator_cls", [SCVIValidator, GeneformerValidator])
def test_validator_with_valid_data_human(validator_cls, loaded_dataset_human):
    v = validator_cls()
    loaded_dataset_human.validate()
    v.validate_dataset(loaded_dataset_human)


def test_validator_raises_with_wrong_dataset_type():
    class WrongDataset(BaseDataset):
        def _validate(self):
            pass

        def load_data(self):
            pass

        def unload_data(self):
            pass

    d = WrongDataset(path="some_path")
    v = UCEValidator()
    with pytest.raises(ValueError):
        v.validate_dataset(d)


@pytest.mark.parametrize(
    "validator_cls",
    [
        UCEValidator,
        SCVIValidator,
        GeneformerValidator,
        ScGenePTValidator,
        ScGPTValidator,
    ],
)
def test_validator_missing_required_inputs(validator_cls, loaded_dataset_human):
    v = validator_cls()
    loaded_dataset_human._inputs.clear()
    with pytest.raises(ValueError):
        v.validate_dataset(loaded_dataset_human)


@pytest.mark.parametrize(
    "validator_cls,missing_obs",
    [
        (SCVIValidator, ["dataset_id"]),
    ],
)
def test_validator_obs_missing_keys(validator_cls, missing_obs, loaded_dataset_human):
    v = validator_cls()
    for k in missing_obs:
        if k in loaded_dataset_human.adata.obs.columns:
            loaded_dataset_human.adata.obs.drop(columns=[k], inplace=True)
    with pytest.raises(ValueError):
        v.validate_dataset(loaded_dataset_human)


@pytest.mark.parametrize(
    "validator_cls,missing_var",
    [
        (UCEValidator, ["gene_symbol"]),
        (ScGenePTValidator, ["gene_symbol"]),
        (ScGPTValidator, ["gene_symbol"]),
        (GeneformerValidator, ["feature_id"]),
    ],
)
def test_validator_var_missing_keys(validator_cls, missing_var, loaded_dataset_human):
    v = validator_cls()
    for k in missing_var:
        if k in loaded_dataset_human.adata.var.columns:
            loaded_dataset_human.adata.var.drop(columns=[k], inplace=True)
    with pytest.raises(ValueError):
        v.validate_dataset(loaded_dataset_human)
