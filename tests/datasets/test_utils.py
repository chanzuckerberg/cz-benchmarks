from czbenchmarks.datasets import utils


def test_list_available_datasets():
    """Test that list_available_datasets returns a sorted list of dataset names."""
    # Get the list of available datasets
    datasets = utils.list_available_datasets()

    # Verify it's a list
    assert isinstance(datasets, list)

    # Verify it's not empty
    assert len(datasets) > 0

    # Verify it's sorted alphabetically
    assert datasets == sorted(datasets)

    # Verify all elements are strings
    assert all(isinstance(dataset, str) for dataset in datasets)

    # Verify no empty strings
    assert all(len(dataset) > 0 for dataset in datasets)
