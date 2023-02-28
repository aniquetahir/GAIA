import numpy as np
from typing import List

def given(data: List, condition_idxs: List):
    # satisfied_idxs = np.arange(len(condition_idxs))
    l1 = np.where(condition_idxs[0])[0]
    for i in range(1, len(condition_idxs)):
        l2 = np.where(condition_idxs[i])[0]
        l1 = np.intersect1d(l1, l2)
    return data[l1]
