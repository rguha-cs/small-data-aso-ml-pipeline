import numpy as np
import pandas as pd
from itertools import combinations

def build_pairwise_diffs(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Returns X_pair (n_pairs, n_features), y_pair (±1), pair_index (list of (i,j))
    Only builds pairs within the same group (e.g., gene).
    """
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)

    X_list, y_list, pairs = [], [], []

    for grp, idx in g.groupby(g).groups.items():
        idx = list(sorted(idx))
        if len(idx) < 2:
            continue
        for i, j in combinations(idx, 2):
            yi, yj = float(y[i]), float(y[j])
            if np.isfinite(yi) and np.isfinite(yj):
                # Convention: target = sign(yi - yj). If higher KD should be ranked higher, keep as is.
                # If lower KD is “better”, flip sign here accordingly.
                s = np.sign(yi - yj)
                if s == 0:
                    continue

                xdiff = (X.iloc[i].values - X.iloc[j].values)
                X_list.append(xdiff)
                y_list.append(s)
                pairs.append((i, j))
                # also add the inverse pair for symmetry
                X_list.append(-xdiff)
                y_list.append(-s)
                pairs.append((j, i))

    if not X_list:
        return None, None, []

    X_pair = np.vstack(X_list)
    y_pair = np.asarray(y_list, dtype=int)
    return X_pair, y_pair, pairs
