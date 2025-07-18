class CZBenchmarksException(Exception):
    """
    Base class for all exceptions in the czbenchmarks package.

    This exception serves as the root for all custom exceptions defined in the
    czbenchmarks package, allowing for consistent error handling.
    """

    pass


class RemoteStorageError(CZBenchmarksException):
    """
    Exception raised for errors related to remote storage.

    This exception is used to indicate issues such as connectivity problems,
    invalid configurations, or other failures when interacting with remote storage.
    """

    pass


class RemoteStorageObjectAlreadyExists(RemoteStorageError):
    """
    Exception raised when attempting to overwrite an existing object in remote storage.

    This exception is triggered when a remote storage operation fails due to
    the target object already existing and overwriting is not permitted.
    """

    pass
