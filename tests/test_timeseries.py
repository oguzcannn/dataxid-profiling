from __future__ import annotations

import math
import random

import polars as pl

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import infer_types
from dataxid_profiling._timeseries import compute_timeseries, seasonality_test, stationarity_test

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