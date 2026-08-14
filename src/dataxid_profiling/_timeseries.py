from __future__ import annotations

import polars as pl
from statsmodels.tsa.stattools import adfuller

from dataxid_profiling._config import ProfileConfig

def compute_timeseries(
    df: pl.DataFrame,
    column_types: dict,
    config: ProfileConfig | None = None,
) -> dict[str, dict]:
    """
    Run stationarity_test on every column inferred as TIMESERIES.

    Returns
    -------
    dict mapping column name → stationarity_test() result.
    """

    if config is None:
        config = ProfileConfig()

    from dataxid_profiling._type_inference import ColumnType

    results: dict[str, dict] = {}

    for col_name, col_type in column_types.items():
        if col_type != ColumnType.TIMESERIES:
            continue

        results[col_name] = stationarity_test(df[col_name], config)

    return results


def stationarity_test(
    series: pl.Series,
    config: ProfileConfig | None = None,
) -> dict:
    """
    Augmented Dickey-Fuller stationarity test for a single numeric series.

    Returns
    -------
    dict with:
        statistic: the ADF test statistic
        p_value: the test's p-value
        is_stationary: True if p_value < config.timeseries_significance
    """

    if config is None:
        config = ProfileConfig()

    values = series.drop_nulls().to_numpy()

    if len(values) < 2:
        return {"statistic": None, "p_value": None, "is_stationary": None}

    statistic, p_value = adfuller(values)[:2]

    return {
        "statistic": statistic,
        "p_value": p_value,
        "is_stationary": bool(p_value < config.timeseries_significance),
    }