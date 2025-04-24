import json
from copy import deepcopy
from datetime import datetime
from functools import wraps
from hashlib import sha256
from typing import Callable

import pandas as pd
from sklearn import clone


def _clone(estimator):
    """
    Create and return a clone of the input estimator.

    Parameters
    ----------
    estimator : object
        The estimator object to be cloned.

    Returns
    -------
    object
        A cloned copy of the input estimator.

    Notes
    -----
    This function attempts to create a clone of the input estimator using the
    `clone` function. If the `clone` function is not available or raises a
    `TypeError`, it falls back to using `deepcopy`. If both methods fail, the
    original estimator is returned.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> original_estimator = LinearRegression()
    >>> cloned_estimator = _clone(original_estimator)
    """
    try:
        return clone(estimator)
    except TypeError:
        pass
    try:
        return deepcopy(estimator)
    except TypeError:
        pass
    return estimator


def get_splitting_matrix(
        X: pd.DataFrame, iter_cross_validation: iter, expand=False
) -> pd.DataFrame:
    """
    Generate a splitting matrix based on cross-validation iterations.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe.
    iter_cross_validation : Iterable
        An iterable containing cross-validation splits (train, test).
    expand : bool, optional
        If True, the output matrix will have columns for both train and test
        splits for each iteration. If False (default), the output matrix will
        have columns for each iteration with 1 for train and 2 for test.

    Returns
    -------
    pd.DataFrame
        A matrix indicating the train (1) and test (2) splits for each
        iteration. Rows represent data points, and columns represent iterations.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
    ...                   'feature2': ['A', 'B', 'C', 'D', 'E']})
    >>> iter_cv = [(range(3), range(3, 5)), (range(2), range(2, 5))]
    >>> get_splitting_matrix(X, iter_cv)
    """
    if not expand:
        all_splits = pd.DataFrame(
            0, index=range(len(X)), columns=range(len(iter_cross_validation))
        )
        for i, (train, test) in enumerate(iter_cross_validation):
            all_splits.loc[train, i] = 1
            all_splits.loc[test, i] = 2
    else:
        all_splits = pd.DataFrame(
            False, index=range(len(X)), columns=range(2 * len(iter_cross_validation))
        )
        for i, (train, test) in enumerate(iter_cross_validation):
            all_splits.loc[train, 2 * i] = True
            all_splits.loc[test, 2 * i + 1] = True

    return all_splits


def get_hash(**kwargs) -> str:
    """Return a hash of parameters"""

    hash_ = sha256()
    for key, value in kwargs.items():
        if isinstance(value, datetime):
            hash_.update(str(kwargs[key]).encode("utf-8"))
        else:
            hash_.update(json.dumps(kwargs[key]).encode())

    return hash_.hexdigest()


def check_started(message: str, need_build: bool = False) -> Callable:
    """
    check_built is a decorator used for methods that must be called on \
    built or unbuilt :class:`~palma.Project`.
    If the :class:`~palma.Project` is_built attribute has \
    not the correct value, an AttributeError is raised with the message passed \
    as argument.

    Parameters
    ----------
    message: str
        Error message
    need_build: bool
        Expected value for :class:`~palma.Project` is_built \
        attribute

    Returns
    -------
    Callable
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(project: "Project", *args, **kwargs) -> Callable:
            if project.is_started == need_build:
                return func(project, *args, **kwargs)
            else:
                raise AttributeError(message)

        return wrapper

    return decorator
