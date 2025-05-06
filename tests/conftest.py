import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--mock-container-runner",
        action="store_true",
        default=False,
        help="Mock the ContainerRunner to avoid nvidia runtime errors if it's not available during testing"
    )

@pytest.fixture
def mock_container_runner(request):
    return request.config.getoption("--mock-container-runner", default=False) 