import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tolerance-percent",
        type=float,
        default=0.2,
        help="Percentage tolerance for metric comparison (default: 0.2 = 20%)",
    )
    parser.addoption(
        "--run-model-tests",
        action="store_true",
        default=False,
        help="Run model regression tests",
    )


def pytest_configure(config):
    pytest.run_model_tests = config.getoption("--run-model-tests")


@pytest.fixture
def tolerance_percent(request):
    return request.config.getoption("--tolerance-percent", default=0.2)


@pytest.fixture
def run_model_tests(request):
    return request.config.getoption("--run-model-tests", default=False)
