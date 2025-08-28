"""
This module defines types used for representing cell data in computational biology tasks.

Types:
    CellRepresentation: A union type representing various formats of cell data,
    including dense arrays, sparse matrices, and dataframes. This type can be
    expanded to include subtypes such as GeneExpression, Embedding, CellImage, etc.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

# TODO: may want to expand this to include subtypes: GeneExpression, Embedding, CellImage, etc.
CellRepresentation = np.ndarray | sp.csr_matrix | pd.DataFrame
