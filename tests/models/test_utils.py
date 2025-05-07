from collections import namedtuple

from czbenchmarks.models import utils


def test_list_available_models():
    """Test that list_available_models returns a sorted list of model names."""
    # Get the list of available datasets
    models = utils.list_available_models()

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


def test_model_to_display_name():
    """Test that we generate model variant display names properly"""
    DummyModelType = namedtuple("DummyModelType", ["name"])
    dummy_model_type = DummyModelType("TESTER")

    # test that the formulaic formatting works
    assert "Tester" == utils.model_to_display_name(dummy_model_type, {})
    assert "Tester" == utils.model_to_display_name(
        dummy_model_type, {"model_variant": "A1"}
    )
    assert "Tester" == utils.model_to_display_name(
        dummy_model_type, {"model_variant": "A1", "dataset": "good_cells"}
    )

    # test that special case lookup works too
    for k, v in utils._MODEL_VARIANT_FINETUNE_TO_DISPLAY_NAME.items():
        dummy_model_type = DummyModelType(k[0])
        args = {"model_variant": k[1]}
        if k[2] is not None:
            args["dataset"] = k[2]
        assert v == utils.model_to_display_name(dummy_model_type, args)
