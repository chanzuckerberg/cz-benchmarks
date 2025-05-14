class CZBenchmarksException(Exception):
    """base class for all exceptions in the czbenchmarks package"""

    pass


class RemoteStorageError(CZBenchmarksException):
    """errors having to do with remote storage"""

    pass


class RemoteStorageObjectAlreadyExists(RemoteStorageError):
    """error when trying to overwrite an already existing object in remote storage"""

    pass
