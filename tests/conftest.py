import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--tolerance-percent",
        type=float,
        default=0.2,
        help="Percentage tolerance for metric comparison (default: 0.2 = 20%)"
    )

@pytest.fixture
def tolerance_percent(request):
    return request.config.getoption("--tolerance-percent", default=0.2)
