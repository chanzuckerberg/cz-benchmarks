import pytest


@pytest.fixture
def setup_context():
    """
    fixture to setup context for test
    """
    return asset


def test_example(setup_context):
    """
    test something
    """
    # assert
