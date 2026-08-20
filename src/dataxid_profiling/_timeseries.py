from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

import numpy as np
from scipy.fft import fft as _fft
from scipy.signal import find_peaks
from dataxid_profiling._config import ProfileConfig

def compute_timeseries(
    df: pl.DataFrame,
    column_types: dict,
    config: ProfileConfig | None = None,
) -> dict[str, dict]:
    """
    Run stationarity_test and seasonality_test on every column inferred as TIMESERIES.
    """

    if config is None:
        config = ProfileConfig()

    from dataxid_profiling._type_inference import ColumnType

    ts_columns = [
        col_name for col_name, col_type in column_types.items()
        if col_type == ColumnType.TIMESERIES
    ]

    if not ts_columns:
        return {}

    if len(ts_columns) == 1:
        col_name = ts_columns[0]
        return {col_name: _analyze_timeseries_column(df[col_name].to_numpy(), config)}

    results: dict[str, dict] = {}
    max_workers = min(len(ts_columns), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_analyze_timeseries_column, df[col_name].to_numpy(), config): col_name
            for col_name in ts_columns
        }
        for future in as_completed(futures):
            col_name = futures[future]
            results[col_name] = future.result()

    return {col: results[col] for col in ts_columns}

def _analyze_timeseries_column(values: np.ndarray, config: ProfileConfig) -> dict:
    series = pl.Series(values)

    stationarity = stationarity_test(series, config)
    seasonality = seasonality_test(series, config)
    curve = acf_pacf_curve(series, config)

    is_stationary = stationarity["is_stationary"]
    is_seasonal = seasonality["seasonality_presence"]

    if is_stationary is not None:
        is_stationary = bool(is_stationary and not is_seasonal)

    return {
        **stationarity,
        "is_stationary": is_stationary,
        **seasonality,
        **curve,
    }

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

    max_allowed_lag = max(0, len(values) // 2 - 2)
    if max_allowed_lag < 1:
        return {"statistic": None, "p_value": None, "is_stationary": None}

    maxlag = min(20, max_allowed_lag)

    try:
        statistic, p_value, *_ = adfuller(
            values,
            autolag="AIC",
            maxlag=maxlag,
        )
    except ValueError:
        return {"statistic": None, "p_value": None, "is_stationary": None}

    return {
        "statistic": statistic,
        "p_value": p_value,
        "is_stationary": bool(p_value < config.timeseries_significance),
    }


def _fft_freq(n: int, d: float = 1.0) -> np.ndarray:
    """FFT sample frequencies, same layout as numpy.fft.fftfreq."""
    val = 1.0 / (n * d)
    results = np.empty(n, dtype=int)
    half = (n - 1) // 2 + 1
    results[:half] = np.arange(0, half, dtype=int)
    results[half:] = np.arange(-(n // 2), 0, dtype=int)
    return results * val


def _compute_fft_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the positive-frequency spectrum as (freq, amplitude_db)."""
    data_fft = _fft(values)
    power = np.abs(data_fft) ** 2
    freqs = _fft_freq(len(power), 1.0)

    pos_mask = freqs > 0
    freq = freqs[pos_mask]
    ampl = 10 * np.log10(power[pos_mask])

    return freq, ampl


def _find_seasonal_peaks(
    freq: np.ndarray,
    ampl: np.ndarray,
    mad_threshold: float,
) -> list[float]:
    """Find dominant spectrum peaks, filtering out harmonics of a base peak."""
    positive_ampl = ampl[ampl > 0]
    if len(positive_ampl) == 0:
        return []

    median = np.median(positive_ampl)
    above_median = positive_ampl[positive_ampl > median]
    mad = 0.0 if len(above_median) == 0 else np.mean(np.abs(above_median - above_median.mean()))

    threshold = median + mad * mad_threshold

    peak_idx, _ = find_peaks(ampl, threshold=0.1)
    if len(peak_idx) == 0:
        return []

    peak_freqs = freq[peak_idx]
    peak_ampls = ampl[peak_idx]

    keep = peak_ampls > threshold
    peak_freqs = peak_freqs[keep]

    if len(peak_freqs) == 0:
        return []

    order = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[order]

    removed = np.zeros(len(peak_freqs), dtype=bool)
    for i in range(len(peak_freqs)):
        if removed[i]:
            continue
        base = peak_freqs[i]
        for j in range(i + 1, len(peak_freqs)):
            if removed[j]:
                continue
            fraction = (peak_freqs[j] / base) % 1
            if fraction < 0.01 or fraction > 0.99:
                removed[j] = True

    return peak_freqs[~removed].tolist()


def seasonality_test(
    series: pl.Series,
    config: ProfileConfig | None = None,
) -> dict:
    """
    FFT-based seasonality detection for a single numeric series.

    Returns
    -------
    dict with:
        seasonality_presence: True if a dominant, non-harmonic peak was found
        seasonalities: estimated periods (1 / frequency), longest first
    """

    if config is None:
        config = ProfileConfig()

    values = series.drop_nulls().to_numpy()

    if len(values) < 4:
        return {"seasonality_presence": False, "seasonalities": []}

    freq, ampl = _compute_fft_spectrum(values)
    peak_freqs = _find_seasonal_peaks(
        freq, ampl, config.timeseries_seasonality_mad_threshold,
    )

    seasonalities = sorted((1.0 / f for f in peak_freqs if f != 0), reverse=True)

    return {
        "seasonality_presence": len(seasonalities) > 0,
        "seasonalities": seasonalities,
    }


def acf_pacf_curve(
    series: pl.Series,
    config: ProfileConfig | None = None,
) -> dict:
    """
    Full ACF/PACF curve for a single numeric series, up to config.timeseries_acf_pacf_lag.
    """

    if config is None:
        config = ProfileConfig()

    values = series.drop_nulls().to_numpy()
    n = len(values)

    if n < 4:
        return {"acf": [], "pacf": []}

    max_lag = min(config.timeseries_acf_pacf_lag, n // 2 - 1)
    if max_lag < 1:
        return {"acf": [], "pacf": []}

    acf_values = acf(values, nlags=max_lag, fft=True)
    pacf_values = pacf(values, nlags=max_lag)

    return {
        "acf": [float(v) for v in acf_values],
        "pacf": [float(v) for v in pacf_values],
    }


def datetime_axis_summary(
    series: pl.Series,
    config: ProfileConfig | None = None,
) -> dict:
    """
    Start/end/typical interval and gap detection for a datetime column.
    """

    if config is None:
        config = ProfileConfig()

    values = series.drop_nulls().sort()

    if len(values) < 2:
        return {
            "start": None, "end": None, "mean_interval_seconds": None,
            "n_gaps": 0, "gap_min_seconds": None,
            "gap_max_seconds": None, "gap_mean_seconds": None,
        }

    diffs = values.diff().drop_nulls().dt.total_seconds().to_numpy()
    mean_interval = float(diffs.mean())

    threshold = config.timeseries_gap_tolerance * mean_interval
    gap_diffs = diffs[diffs > threshold]

    return {
        "start": values[0],
        "end": values[-1],
        "mean_interval_seconds": mean_interval,
        "n_gaps": int(len(gap_diffs)),
        "gap_min_seconds": float(gap_diffs.min()) if len(gap_diffs) else None,
        "gap_max_seconds": float(gap_diffs.max()) if len(gap_diffs) else None,
        "gap_mean_seconds": float(gap_diffs.mean()) if len(gap_diffs) else None,
    }


def compute_datetime_summary(
    df: pl.DataFrame,
    column_types: dict,
    config: ProfileConfig | None = None,
) -> dict[str, dict]:
    """
    Run datetime_axis_summary on every column inferred as DATETIME.
    """

    if config is None:
        config = ProfileConfig()

    from dataxid_profiling._type_inference import ColumnType

    results: dict[str, dict] = {}

    for col_name, col_type in column_types.items():
        if col_type != ColumnType.DATETIME:
            continue

        results[col_name] = datetime_axis_summary(df[col_name], config)

    return results