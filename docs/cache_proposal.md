# CZ-Benchmarks Storage and Caching

# Introduction

The cz-benchmarks package needs to be able to make use of large filesize models and datasets stored in a variety of different systems. Currently, the CLI makes use of caching the results of inference to save computation, and it would be nice to continue supporting that. In the long term, cz-benchmarks might need to be able to cache models fine-tuned for particular benchmarking tasks, or checkpoints created during fine-tuning.

# Storage

Datasets, models, and other cached items can potentially be stored in a variety of systems, with different ways of accessing them. For example, a dataset could be stored locally on a machine (for example, in the bring-your-own-dataset use case), in a cloud-based object store (most datasets are currently stored in Amazon S3), or on a networked filesystem (for example, in a cluster).  
Some of these systems have different interfaces (for example, S3 and a local filesystem). Some of these systems have different usage patterns (for example, the networked filesystem may be read only). It will be useful for cz-benchmarks to provide common interfaces for these to support building a caching layer on top.  
That said, I expect we will need to include some special behaviors for the local filesystem. The models I've looked at almost all have places where they expect paths to files, or directories that they access without going through cz-benchmarks. It's also easier to write storage interfaces if we can rely on "copy to local filesystem" and "copy from local filesystem" to always be available.

## Interfaces

Names may change, but this hopefully gives an idea of what we need to support:

```py
class Storage:
    """ every storage system at least needs to implement these """
    def copy(self, key: str, destination: LocalStorage):
        """ copy a file from the storage (e.g. download) to the local storage """
        ...
    def get_hash(self, key: str) -> bytes:
        """ return a SHA(?) hash of the object specified by key, in order to
            allow us to verify files are actually the same (or not)
        """
        ...
    def get_last_modified(self, key: str) -> datetime.datetime:
        """ return a datetime when the file was last modified, in order to
            support caching that can overwrite older files
        """
        ...


class ReadOnlyStorage(Storage):
    ...


class ReadWriteStorage(Storage):
    def copy_from(self, key: str, source: LocalStorage):
        """ copy a file from local storage to the other storage (e.g. upload) """
        ...

    def delete(self, key: str):
       ...

    def copy_from_if_local_is_newer(self, key: str, source: LocalStorage):
        """ This may not be needed in the initial release.
            We may need to support conditionally overwriting older files
        """
        ...

    def copy_from_if_overwriting_hash(self, key: str, hash: bytes, source: LocalStorage):
        """ This may not be needed in the initial release.
            If there are multiple jobs writing to the same storage, we may need
            to support a way to only overwrite files we're sure we want to overwrite
            and not, perhaps, a file that another job updated while our job was
            working
        """

    def delete_if_hash_matches(self, key: str, hash: bytes):
        """ This may not be needed in the initial release.
            Same idea as the previous one, but for deleting.
        """
        ...


class LocalStorage(ReadWriteStorage):
    """ the ususal use case would be for this to be a thin wrapper
        over the usual filesystem, but it could do things like
        name files on disk differently than their keys, or use
        temporary directories that get cleaned up when it is
        garbage collected
    """
    def path(self, key: str) -> pathlib.Path:
        ...

    def open(self, key: str, mode: str = "r") -> io.IOBase:
        ...
```

## Classes to Implement

### Definitely Need

```py
class SimpleLocalStorage(LocalStorage):
    def __init__(self, root_path: str | pathlib.Path):
        """ store files by keys that are the filenames, under some root directory """
        ...

class S3Storage(ReadWriteStorage):
    """ This will support AWS S3, but also anything that supports the S3 protocol,
        which a lot of different cloud storage providers do
        There may need to be separate classes for signed and unsigned
        (public bucket) requests
    """
    def __init__(self, bucket: str, prefix: str, ...):  # things like address for non-AWS
        ...
```

### Nice to Have / More Ideas

```py
class TemporaryLocalStorage(LocalStorage):
    """ Storage that gets cleaned up when the job exits """
    ...

class ReadOnlyFilesystemStorage(ReadOnlyStorage):
    """ for example, maybe all the machines on a cluster read from the same
        network share, but the user doesn't want them writing back to that
        share
    """
    ...

class PartitioningLocalStorage(LocalStorage):
    """ if someone is storing a lot of files, and doesn't want to manage their own
        directory structure, this could handle distributing them between subfolders.
        Almost certainly not needed initially, but this is the type of thing we could
        support with this interface
    """
    ...
```

# Caching

Caching is built on top of storage. For the initial release, we probably need to ultimately supply cached files from some form of `LocalStorage` for models and tasks to use.

## Interface

```py
class Cache:
    def get(self, key: str) -> pathlib.Path:
        """ get a path for the file in the cache, copying it as needed """
        ...

    def open(self, key: str, mode: str = "r") -> io.IOBase:
        """ return a buffer to read from the file specified by key in
            the cache
        """
        ...

    def get_hash(self, key: str) -> bytes:
        """ return a SHA(?) hash of the object specified by key, in order to
            allow us to verify files are actually the same (or not)
        """
        ...

    def get_last_modified(self, key: str) -> datetime.datetime:
        """ return a datetime when the file was last modified, in order to
            support caching that can overwrite older files
        """
        ...

class WriteableCache(Cache):
    def put(self, key: str, file: pathlib.Path):
        """ put the local file into the cache under the specified key,
            copying it as needed
        """
        ...

    def delete(self, key: str, file: pathlib.Path):
        """ put the local file into the cache under the specified key,
            copying it as needed
        """
        ...
```

## Classes To Implement

By using the type system and the different types of storage, these help convey what users can expect to be cached where.

```py
class LocalReadOnlyCache(Cache):
    """ Supports reading from cached files already stored locally """
    def __init__(self, storage: LocalReadOnlyStorage):
        ...

class LocalCache(WriteableCache):
    """ Supports reading from and writing to a local cache """
    def __init__(self, storage: LocalStorage):
        ...

class CopyToLocalCache(Cache):
    """ Copies files locally from other storage systems.
        The other stores will be checked and cache hits will be synced locally.
        This may have different modes, or there may be two similar classes,
        to support things like:
        1) does it always download files from the cache if they're newer than local?
        2) does it use the first cache hit or the newest modified cache hit?
        etc.
        But we don't need all of those covered for the initial release
    """
    def __init__(self, read_storage: list[ReadOnlyStorage], local_storage: LocalStorage):
        ...

class SynchronizedRemoteAndLocalCache(WriteableCache):
    """ Keeps a local cache and a remote cache, copying from the remote
        to the local, or from the local to the remote, whenever the user
        performs operations
    """
    def __init___(self, remote_storage: ReadWriteStorage, local_storage: LocalStorage):
        ...
```

And potentially more but that's a basic set.

# How this works within CZ-Benchmarks

This will probably evolve as we understand better how the components interact, but my thoughts for now are:

## Datasets and Models should have a way to pass in (read only) caching

When a dataset actually needs to download files, or a model needs to download its weights, there should be some way for the user to supply a cache. It might be as simple as `Dataset.load_data(cache: Cache)`. Most of the time this might be some default cache configured to read from public S3 buckets. But this would also allow a user pass in their own cache. For example, if they had a cluster and wanted to host everything internal to their cluster rather than having to keep reaching out over the internet.

## Cache registry / configuration

There should be some set of caches built into the library and available without the user having to do much work. For example, there should at least be:

* a local cache in a standard place (we should be using something based on $`XDG_CACHE_HOME` instead of `~/.cz-benchmarks` however)  
* a remote read-only cache that uses the public S3 buckets

And users should be able to configure their own caches in a (YAML?) config file to use across repeated invocations of cz-benchmarks. I think it would be good if they could name these caches and specify which ones to use by default if not the ones we provide. Then, when running the CLI, they could simply supply a cache name and it would use what they'd configured, rather than having to completely configure caching through CLI arguments.  
