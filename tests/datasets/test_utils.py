from czbenchmarks.datasets import utils


def test_list_available_datasets():
    """Test that list_available_datasets returns a sorted list of dataset names."""
    # Get the list of available datasets
    datasets = utils.list_available_datasets()

    # Verify it's a dict
    assert isinstance(datasets, dict)

    # Verify it's not empty
    assert len(datasets) > 0

    # Verify it's sorted alphabetically
    assert list(datasets.keys()) == sorted(datasets.keys())

    # Verify the dataset names match the expected dataset names
    expected_datasets = {
        "adamson_perturb": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/perturb/single_cell/adamson_perturbation.h5ad",
        },
        "dixit_perturb": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/perturb/single_cell/dixit_perturbation.h5ad",
        },
    }
    assert datasets["adamson_perturb"] == expected_datasets["adamson_perturb"]
    assert datasets["dixit_perturb"] == expected_datasets["dixit_perturb"]
