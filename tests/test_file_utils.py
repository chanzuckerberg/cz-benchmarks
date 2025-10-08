import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from czbenchmarks.file_utils import CacheManager, download_file_from_remote


@pytest.fixture
def temp_file(tmp_path):
    """Fixture to create a temporary file."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Sample content")
    return file_path


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to create a temporary directory."""
    return tmp_path


@pytest.fixture
def cache_manager(temp_dir):
    """Fixture to create a CacheManager instance."""
    return CacheManager(cache_dir=temp_dir, expiration_days=1)


def test_cache_manager_ensure_directory_exists(temp_dir):
    """Test CacheManager.ensure_directory_exists creates the directory."""
    cache_manager = CacheManager(cache_dir=temp_dir / "new_cache_dir")
    cache_manager.ensure_directory_exists(cache_manager.cache_dir)
    assert (temp_dir / "new_cache_dir").exists()


def test_cache_manager_get_cache_path(temp_dir):
    """Test CacheManager.get_cache_path generates correct cache path."""
    cache_manager = CacheManager(cache_dir=temp_dir)
    remote_url = "https://example.com/file.txt"
    expected_path = temp_dir / "file.txt"
    assert cache_manager.get_cache_path(remote_url) == expected_path


def test_cache_manager_is_expired(temp_file, cache_manager):
    """Test CacheManager.is_expired correctly identifies expired files."""

    assert not cache_manager.is_expired(temp_file)

    expired_time = datetime.now() - timedelta(days=2)
    os.utime(temp_file, (expired_time.timestamp(), expired_time.timestamp()))
    assert cache_manager.is_expired(temp_file)


def test_cache_manager_clean_expired_cache(temp_file, cache_manager):
    """Test CacheManager.clean_expired_cache removes expired files."""

    expired_time = datetime.now() - timedelta(days=2)
    os.utime(temp_file, (expired_time.timestamp(), expired_time.timestamp()))

    cache_manager.clean_expired_cache()
    assert not temp_file.exists()


def test_download_file_from_remote(monkeypatch, temp_dir):
    """Test download_file_from_remote caches an S3 file without real download."""

    remote_url = "s3://cz-benchmarks-data/datasets/v2/file.txt"
    mock_client = MagicMock()

    def fake_download(bucket, key, filename):
        Path(filename).write_text("mock-data")

    mock_client.download_file.side_effect = fake_download

    monkeypatch.setattr(
        "czbenchmarks.file_utils._get_s3_client",
        lambda make_unsigned_request=True: mock_client,
    )

    local_path = download_file_from_remote(remote_url, cache_dir=temp_dir)

    expected_path = temp_dir / "file.txt"
    assert Path(local_path) == expected_path
    assert expected_path.exists()
    assert expected_path.read_text() == "mock-data"
    mock_client.download_file.assert_called_once_with(
        "cz-benchmarks-data", "datasets/v2/file.txt", str(expected_path)
    )


@pytest.mark.parametrize("remote_url, exception_match", [
    ("https://example.com/file.txt", "Unsupported protocol"),   
    ("/tmp/local/path/dummy.h5ad", "Only S3 paths are supported, got local path"),
])
def test_download_file_from_remote_invalid_protocol(temp_dir, remote_url, exception_match):
    """Test download_file_from_remote rejects unsupported URLs."""
    with pytest.raises(ValueError, match=exception_match):
        download_file_from_remote(remote_url, cache_dir=temp_dir)