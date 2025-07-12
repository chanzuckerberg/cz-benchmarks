from typing import Union
import numpy as np
import pandas as pd
import scipy.sparse as sp


# TODO: may want to expand this to include subtypes: GeneExpression, Embedding, CellImage, etc.
CellRepresentation = Union[np.ndarray, sp.csr_matrix, pd.DataFrame]
