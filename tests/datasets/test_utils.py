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


def test_dataset_to_display_name():
    """Test that we genererate dataset display names properly"""
    # test that the formulaic formatting works
    assert "My dataset" == utils.dataset_to_display_name("my_dataset")
    assert "Example dataset" == utils.dataset_to_display_name("EXAMPLE_DATASET")
    assert "Short" == utils.dataset_to_display_name("short")
    assert "This is a really long one" == utils.dataset_to_display_name(
        "This_is_A_ReAlLy_loNG_ONE"
    )
    assert "" == utils.dataset_to_display_name("")

    # test that special case lookup works too
    for k, v in utils._DATASET_TO_DISPLAY_NAME.items():
        assert v == utils.dataset_to_display_name(k)
