import numpy as np
import pandas as pd
import scipy.sparse as sp


# TODO: may want to expand this to include subtypes: GeneExpression, Embedding, CellImage, etc.
CellRepresentation = np.ndarray | sp.csr_matrix | pd.DataFrame
