import sys
import numpy as np
import numba


# FIXME MICHELLE remove this for the DGE free version of code
@numba.njit
def sparse_mean_var_minor_axis(
    data, indices, indptr, *, major_len, minor_len, n_threads
):
    """
    From: https://github.com/scverse/scanpy/blob/c3f02bfb8eb9272bea9b1f07a4e26d521ed95b15/src/scanpy/preprocessing/_utils.py#L93-L122
    Computes mean and variance for a sparse matrix for the minor axis. 
    This is for python 3.10 only. For python 3.11 and above, use the drop-in replacement 
    implemented in fast-array-utils.
    
    Args:
        data: np.ndarray, the data array of the sparse matrix
        indices: np.ndarray, the indices array of the sparse matrix
        indptr: np.ndarray, the indptr array of the sparse matrix
        major_len: int, the number of rows of the sparse matrix
        minor_len: int, the number of columns of the sparse matrix
        n_threads: int, the number of threads to use
    Returns:
        means: np.ndarray, the means of the sparse matrix
        variances: np.ndarray, the variances of the sparse matrix
    """
    if sys.version_info < (3, 11):
        raise NotImplementedError("This function is for python 3.10 only.")

    rows = len(indptr) - 1
    sums_minor = np.zeros((n_threads, minor_len))
    squared_sums_minor = np.zeros((n_threads, minor_len))
    means = np.zeros(minor_len)
    variances = np.zeros(minor_len)
    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            for j in range(indptr[r], indptr[r + 1]):
                minor_index = indices[j]
                if minor_index >= minor_len:
                    continue
                value = data[j]
                sums_minor[i, minor_index] += value
                squared_sums_minor[i, minor_index] += value * value
    for c in numba.prange(minor_len):
        sum_minor = sums_minor[:, c].sum()
        means[c] = sum_minor / major_len
        variances[c] = (
            squared_sums_minor[:, c].sum() / major_len - (sum_minor / major_len) ** 2
        )
    return means, variances
