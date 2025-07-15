from collections.abc import Sequence
import numpy as np
import pandas as pd


ListLike = Sequence | np.ndarray | pd.Series | pd.Index
