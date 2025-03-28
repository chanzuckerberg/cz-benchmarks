from czbenchmarks.models.utils import list_available_models


def test_list_available_models():
    """Test that list_available_models returns a sorted list of model names."""
    # Get the list of available datasets
    models = list_available_models()

    # Verify it's a list
    assert isinstance(models, list)

    # Verify it's not empty
    assert len(models) > 0

    # Verify it's sorted alphabetically
    assert models == sorted(models)

    # Verify all elements are strings
    assert all(isinstance(model, str) for model in models)

    # Verify no empty strings
    assert all(len(model) > 0 for model in models)
