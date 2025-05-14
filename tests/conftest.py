import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--tolerance-percent",
        type=float,
        default=0.1,
        help="Percentage tolerance for metric comparison (default: 0.1 = 10%)"
    )

@pytest.fixture
def tolerance_percent(request):
    return request.config.getoption("--tolerance-percent", default=0.1)
