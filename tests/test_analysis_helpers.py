"""Tests for private helper functions in endive/saturation_analysis.py."""

import numpy as np
import pandas as pd
import pytest

from endive.saturation_analysis import (
    _apply_warmup_cooldown,
    _compute_seed_metrics,
    _extract_param_value,
)


# ---------------------------------------------------------------------------
# _extract_param_value
# ---------------------------------------------------------------------------


class TestExtractParamValue:
    """Tests for _extract_param_value(cfg, param_name)."""

    @pytest.fixture()
    def full_cfg(self):
        return {
            "transaction": {
                "inter_arrival": {"scale": 50.0},
                "operation_types": {"fast_append": 0.8},
                "real_conflict_probability": 0.3,
            },
            "catalog": {
                "num_tables": 5,
                "service": {"latency_ms": 12.0},
            },
            "storage": {"provider": "s3x"},
            "simulation": {"duration_ms": 3600000},
        }

    def test_inter_arrival_scale(self, full_cfg):
        assert _extract_param_value(full_cfg, "inter_arrival_scale") == 50.0

    def test_fast_append_ratio(self, full_cfg):
        assert _extract_param_value(full_cfg, "fast_append_ratio") == 0.8

    def test_real_conflict_probability(self, full_cfg):
        assert _extract_param_value(full_cfg, "real_conflict_probability") == 0.3

    def test_num_tables(self, full_cfg):
        assert _extract_param_value(full_cfg, "num_tables") == 5

    def test_catalog_service_latency_ms(self, full_cfg):
        assert _extract_param_value(full_cfg, "catalog_service_latency_ms") == 12.0

    def test_storage_provider(self, full_cfg):
        assert _extract_param_value(full_cfg, "storage_provider") == "s3x"

    def test_fallback_to_section_lookup(self, full_cfg):
        """A param not in param_map should be found via direct section scan."""
        assert _extract_param_value(full_cfg, "duration_ms") == 3600000

    def test_missing_key_returns_none(self, full_cfg):
        assert _extract_param_value(full_cfg, "nonexistent_param") is None

    def test_empty_config_returns_none(self):
        assert _extract_param_value({}, "num_tables") is None

    def test_partial_config_missing_nested_key(self):
        """inter_arrival_scale when transaction section has no inter_arrival."""
        cfg = {"transaction": {"retry": 5}}
        assert _extract_param_value(cfg, "inter_arrival_scale") is None


# ---------------------------------------------------------------------------
# _compute_seed_metrics
# ---------------------------------------------------------------------------


def _make_txn_df(
    n: int = 100,
    status: str = "committed",
    t_start: float = 0.0,
    t_end: float = 3600000.0,
    latency: float = 100.0,
    operation_type: str = None,
):
    """Build a minimal transaction DataFrame for testing.

    Transactions are uniformly spaced across [t_start, t_end].
    """
    t_submit = np.linspace(t_start, t_end, n, endpoint=False)
    t_commit = t_submit + latency
    data = {
        "t_submit": t_submit,
        "t_commit": t_commit,
        "commit_latency": np.full(n, latency),
        "status": [status] * n,
    }
    if operation_type is not None:
        data["operation_type"] = [operation_type] * n
    return pd.DataFrame(data)


class TestComputeSeedMetrics:
    """Tests for _compute_seed_metrics(df, op_type_filter)."""

    def test_all_committed_returns_metrics(self):
        df = _make_txn_df(n=200, latency=100.0)
        result = _compute_seed_metrics(df)
        assert result is not None
        assert result["success_rate"] == 100.0
        assert result["mean_latency"] == 100.0
        assert result["p50_latency"] == 100.0
        assert result["throughput"] > 0

    def test_all_aborted_returns_none(self):
        df = _make_txn_df(n=50, status="aborted")
        result = _compute_seed_metrics(df)
        assert result is None

    def test_mixed_status_success_rate(self):
        """50 committed + 50 aborted should yield ~50% success_rate."""
        df_c = _make_txn_df(n=50, status="committed", t_start=0, t_end=1800000)
        df_a = _make_txn_df(n=50, status="aborted", t_start=1800000, t_end=3600000)
        df = pd.concat([df_c, df_a], ignore_index=True)
        result = _compute_seed_metrics(df)
        # success_rate is computed over ALL transactions (not just steady-state)
        assert result is not None
        assert result["success_rate"] == pytest.approx(50.0)

    def test_op_type_filter_selects_matching(self):
        df_fa = _make_txn_df(n=80, operation_type="fast_append")
        df_vo = _make_txn_df(
            n=20, operation_type="validated_overwrite", latency=200.0
        )
        df = pd.concat([df_fa, df_vo], ignore_index=True)

        result = _compute_seed_metrics(df, op_type_filter="validated_overwrite")
        assert result is not None
        assert result["mean_latency"] == 200.0

    def test_op_type_filter_no_match_returns_none(self):
        df = _make_txn_df(n=50, operation_type="fast_append")
        result = _compute_seed_metrics(df, op_type_filter="validated_overwrite")
        assert result is None

    def test_op_type_filter_missing_column_returns_none(self):
        df = _make_txn_df(n=50)  # no operation_type column
        result = _compute_seed_metrics(df, op_type_filter="fast_append")
        assert result is None

    def test_latency_percentiles(self):
        """With varying latencies, percentiles should differ."""
        n = 1000
        rng = np.random.RandomState(42)
        latencies = rng.exponential(scale=100.0, size=n)
        t_submit = np.linspace(0, 3600000, n, endpoint=False)
        t_commit = t_submit + latencies
        df = pd.DataFrame({
            "t_submit": t_submit,
            "t_commit": t_commit,
            "commit_latency": latencies,
            "status": ["committed"] * n,
        })
        result = _compute_seed_metrics(df)
        assert result is not None
        assert result["p50_latency"] < result["p95_latency"]
        assert result["p95_latency"] < result["p99_latency"]


# ---------------------------------------------------------------------------
# _apply_warmup_cooldown
# ---------------------------------------------------------------------------


class TestApplyWarmupCooldown:
    """Tests for _apply_warmup_cooldown(df)."""

    def test_middle_80_percent_kept(self):
        """With uniform spacing over [0, 1000], the 10% warmup/cooldown should
        trim transactions whose t_submit < 100 or t_commit > 900."""
        n = 100
        t_submit = np.linspace(0, 990, n)
        latency = 5.0
        t_commit = t_submit + latency
        df = pd.DataFrame({
            "t_submit": t_submit,
            "t_commit": t_commit,
        })
        # duration = t_commit.max() - t_submit.min() = 995
        # warmup = 99.5, so t_min = 99.5, t_max = 995 - 99.5 = 895.5
        result = _apply_warmup_cooldown(df)
        # Boundary transactions with t_submit < 99.5 are excluded
        assert result["t_submit"].min() >= 99.5
        # Boundary transactions with t_commit > 895.5 are excluded
        assert result["t_commit"].max() <= 895.5
        # Some rows should remain
        assert len(result) > 0
        assert len(result) < n

    def test_zero_duration_returns_full_df(self):
        """All identical timestamps means duration=0, should return all rows."""
        df = pd.DataFrame({
            "t_submit": [100.0, 100.0, 100.0],
            "t_commit": [100.0, 100.0, 100.0],
        })
        result = _apply_warmup_cooldown(df)
        assert len(result) == 3

    def test_preserves_columns(self):
        """Extra columns should survive the filter."""
        df = pd.DataFrame({
            "t_submit": np.linspace(0, 1000, 50),
            "t_commit": np.linspace(10, 1010, 50),
            "status": ["committed"] * 50,
            "extra": range(50),
        })
        result = _apply_warmup_cooldown(df)
        assert "status" in result.columns
        assert "extra" in result.columns
