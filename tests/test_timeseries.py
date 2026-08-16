from __future__ import annotations

import math
import random

import polars as pl

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import infer_types
from dataxid_profiling._timeseries import acf_pacf_curve, compute_timeseries, seasonality_test, stationarity_test, datetime_axis_summary

import pytest

class TestStationarityTest:
    def test_trending_series_is_not_stationary(self):
        rng = random.Random(42)
        n = 200
        trend = pl.Series([i * 0.5 + rng.gauss(0, 1) for i in range(n)])

        result = stationarity_test(trend)

        assert result["is_stationary"] is False

    def test_stationary_series_is_detected(self):
        rng = random.Random(42)
        n = 200
        # Pure noise around a fixed mean has no trend → should be stationary.
        stationary = pl.Series([rng.gauss(0, 1) for _ in range(n)])

        result = stationarity_test(stationary)

        assert result["is_stationary"] is True

    def test_too_few_values_returns_none(self):
        result = stationarity_test(pl.Series([1.0]))

        assert result["is_stationary"] is None


class TestComputeTimeseries:
    def test_only_timeseries_columns_are_included(self):
        rng = random.Random(42)
        n = 200
        trend = [i * 0.5 + rng.gauss(0, 1) for i in range(n)]
        noise = [rng.gauss(0, 1) for _ in range(n)]

        df = pl.DataFrame({"trend_col": trend, "random_col": noise})
        config = ProfileConfig(timeseries_active=True)

        column_types = infer_types(df, config)
        result = compute_timeseries(df, column_types, config)

        assert "trend_col" in result
        assert "random_col" not in result

    def test_empty_when_timeseries_inactive(self):
        rng = random.Random(42)
        n = 200
        trend = [i * 0.5 + rng.gauss(0, 1) for i in range(n)]

        df = pl.DataFrame({"trend_col": trend})
        config = ProfileConfig()  # timeseries_active defaults to False

        column_types = infer_types(df, config)
        result = compute_timeseries(df, column_types, config)

        assert result == {}


class TestSeasonalityTest:
    def test_seasonal_series_is_detected(self):
        n = 200
        seasonal = [10 * math.sin(2 * math.pi * i / 7) for i in range(n)]

        result = seasonality_test(pl.Series(seasonal))

        assert result["seasonality_presence"] is True
        assert len(result["seasonalities"]) > 0

    def test_trend_only_series_has_no_seasonality(self):
        n = 200
        trend = [i * 0.5 for i in range(n)]

        result = seasonality_test(pl.Series(trend))

        assert result["seasonality_presence"] is False
        assert result["seasonalities"] == []

    def test_too_few_values_returns_no_seasonality(self):
        result = seasonality_test(pl.Series([1.0, 2.0]))

        assert result["seasonality_presence"] is False


class TestComputeTimeseriesWithSeasonality:
    def test_seasonal_result_included(self):
        n = 200
        seasonal = [10 * math.sin(2 * math.pi * i / 7) for i in range(n)]

        df = pl.DataFrame({"seasonal_col": seasonal})
        config = ProfileConfig(timeseries_active=True)

        column_types = infer_types(df, config)
        result = compute_timeseries(df, column_types, config)

        assert "seasonal_col" in result
        assert result["seasonal_col"]["seasonality_presence"] is True


class TestAcfPacfCurve:
    def test_returns_requested_length(self):
        n = 200
        seasonal = [10 * math.sin(2 * math.pi * i / 7) for i in range(n)]

        config = ProfileConfig(timeseries_acf_pacf_lag=100)
        result = acf_pacf_curve(pl.Series(seasonal), config)

        assert len(result["acf"]) == 100  # capped by n // 2 - 1 = 99, so 0..99
        assert len(result["pacf"]) == 100

    def test_lag_zero_is_always_one(self):
        n = 200
        seasonal = [10 * math.sin(2 * math.pi * i / 7) for i in range(n)]

        result = acf_pacf_curve(pl.Series(seasonal))

        assert result["acf"][0] == pytest.approx(1.0)

    def test_too_few_values_returns_empty(self):
        result = acf_pacf_curve(pl.Series([1.0, 2.0]))

        assert result["acf"] == []
        assert result["pacf"] == []


class TestDatetimeAxisSummary:
    def test_detects_start_and_end(self):
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        result = datetime_axis_summary(pl.Series(dates))

        assert result["start"] == dates[0]
        assert result["end"] == dates[-1]

    def test_detects_gap(self):
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        dates += [datetime(2024, 1, 20) + timedelta(days=i) for i in range(10)]

        result = datetime_axis_summary(pl.Series(dates))

        assert result["n_gaps"] == 1
        assert result["gap_mean_seconds"] == pytest.approx(864000.0)

    def test_no_gap_when_regular(self):
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        result = datetime_axis_summary(pl.Series(dates))

        assert result["n_gaps"] == 0

    def test_too_few_values_returns_none(self):
        from datetime import datetime

        result = datetime_axis_summary(pl.Series([datetime(2024, 1, 1)]))

        assert result["start"] is None
        assert result["n_gaps"] == 0