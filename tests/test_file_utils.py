import pytest
import os
from datetime import datetime, timedelta
from czbenchmarks.file_utils import CacheManager, upload_file_to_remote


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
    # File should not be expired immediately after creation
    assert not cache_manager.is_expired(temp_file)

    # Simulate expiration by modifying the file's timestamp
    expired_time = datetime.now() - timedelta(days=2)
    os.utime(temp_file, (expired_time.timestamp(), expired_time.timestamp()))
    assert cache_manager.is_expired(temp_file)


def test_cache_manager_clean_expired_cache(temp_file, cache_manager):
    """Test CacheManager.clean_expired_cache removes expired files."""
    # Simulate expiration
    expired_time = datetime.now() - timedelta(days=2)
    os.utime(temp_file, (expired_time.timestamp(), expired_time.timestamp()))

    cache_manager.clean_expired_cache()
    assert not temp_file.exists()


def test_upload_file_to_remote_invalid_remote_url(temp_file):
    """Test upload_file_to_remote raises ValueError for invalid remote prefix URL."""
    invalid_remote_prefix_url = "invalid_prefix"
    with pytest.raises(
        ValueError,
        match=f"Remote URL {invalid_remote_prefix_url!r} should be a prefix ending in '/'",
    ):
        upload_file_to_remote(str(temp_file), invalid_remote_prefix_url)
