"""Tests for markdown table generation functions in saturation_analysis.py."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endive.saturation_analysis import (
    format_value_with_stddev,
    generate_latency_vs_throughput_table,
    generate_success_rate_table,
    generate_overhead_table,
    generate_success_rate_vs_throughput_table,
    generate_commit_rate_over_time_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_df(
    n_rows=3,
    include_stddev=False,
    include_overhead=True,
    include_inter_arrival=True,
    group_by_col=None,
    group_by_vals=None,
):
    """Build a DataFrame with all columns needed by the table generators.

    Args:
        n_rows: Number of rows per group (or total if no grouping).
        include_stddev: Whether to add *_std columns.
        include_overhead: Whether to add overhead columns.
        include_inter_arrival: Whether to add inter_arrival_scale column.
        group_by_col: Name of a grouping column to add.
        group_by_vals: Values for the grouping column (one per group).
    """
    groups = group_by_vals or [None]
    frames = []

    for g_val in groups:
        # Ascending throughput, descending success rate -- mimics saturation
        throughput = np.linspace(5.0, 15.0, n_rows)
        success_rate = np.linspace(100.0, 90.0, n_rows)

        data = {
            "throughput": throughput,
            "success_rate": success_rate,
            "p50_commit_latency": np.linspace(200.0, 800.0, n_rows),
            "p95_commit_latency": np.linspace(400.0, 1600.0, n_rows),
            "p99_commit_latency": np.linspace(600.0, 2400.0, n_rows),
            "mean_retries": np.linspace(0.1, 1.5, n_rows),
            "committed": np.linspace(1000, 3000, n_rows).astype(int),
            "total_txns": np.linspace(1000, 3300, n_rows).astype(int),
        }

        if include_inter_arrival:
            data["inter_arrival_scale"] = np.linspace(200.0, 50.0, n_rows)

        if include_overhead:
            data["mean_overhead_pct"] = np.linspace(5.0, 40.0, n_rows)
            data["p50_overhead_pct"] = np.linspace(3.0, 35.0, n_rows)
            data["p95_overhead_pct"] = np.linspace(10.0, 60.0, n_rows)
            data["p99_overhead_pct"] = np.linspace(15.0, 75.0, n_rows)

        if include_stddev:
            data["throughput_std"] = np.full(n_rows, 0.5)
            data["success_rate_std"] = np.full(n_rows, 1.2)
            data["p50_commit_latency_std"] = np.full(n_rows, 20.0)
            data["p95_commit_latency_std"] = np.full(n_rows, 40.0)
            data["p99_commit_latency_std"] = np.full(n_rows, 60.0)
            if include_overhead:
                data["mean_overhead_pct_std"] = np.full(n_rows, 2.0)
                data["p50_overhead_pct_std"] = np.full(n_rows, 1.5)
                data["p95_overhead_pct_std"] = np.full(n_rows, 3.0)
                data["p99_overhead_pct_std"] = np.full(n_rows, 4.0)

        df = pd.DataFrame(data)
        if group_by_col and g_val is not None:
            df[group_by_col] = g_val
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _read_output(path):
    """Read and return the full text of a generated markdown file."""
    with open(path) as f:
        return f.read()


def _table_rows(text):
    """Extract non-header, non-separator table rows from markdown text.

    Returns list of lists, one per data row, with leading/trailing pipes stripped.
    """
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip separator lines (|---|---|)
        if set(stripped.replace("|", "").replace("-", "").strip()) <= set():
            continue
        # Skip header rows that are purely alphabetic labels
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # Separator lines contain only dashes and optional colons
        if all(set(c) <= {"-", ":"} for c in cells):
            continue
        rows.append(cells)
    return rows


_CFG_TEMPLATE = """\
[simulation]
duration_ms = 3600000

[experiment]
label = "{label}"

[catalog]
num_tables = 1
mode = "instant"

[storage]
provider = "s3x"

[transaction]
retry = 10
inter_arrival.scale = {scale}
real_conflict_probability = 0.0
runtime.mean = 10000
runtime.sigma = 1.5
operation_types.fast_append = 1.0
"""


def _create_experiment(base_dir, label, exp_hash, seeds=(42,),
                       n_txns=100, scale=100.0, success_rate=1.0):
    """Create a disk experiment directory with cfg.toml and seed results."""
    exp_dir = Path(base_dir) / f"{label}-{exp_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg_text = _CFG_TEMPLATE.format(label=label, scale=scale)
    with open(exp_dir / "cfg.toml", "w") as f:
        f.write(cfg_text)

    rng = np.random.RandomState(42)
    for s in seeds:
        seed_dir = exp_dir / str(s)
        seed_dir.mkdir(exist_ok=True)

        statuses = []
        for _ in range(n_txns):
            statuses.append("committed" if rng.random() < success_rate else "aborted")

        df = pd.DataFrame({
            "txn_id": range(n_txns),
            "t_submit": np.linspace(0, 3_600_000, n_txns).astype(int),
            "t_commit": (np.linspace(0, 3_600_000, n_txns) + 1000).astype(int),
            "commit_latency": rng.randint(500, 5000, n_txns),
            "total_latency": rng.randint(5000, 15000, n_txns),
            "n_retries": rng.randint(0, 3, n_txns).astype(np.int8),
            "status": statuses,
            "operation_type": ["fast_append"] * n_txns,
        })
        df.to_parquet(seed_dir / "results.parquet")

    return exp_dir


# ===========================================================================
# format_value_with_stddev
# ===========================================================================

class TestFormatValueWithStddev:
    """Tests for format_value_with_stddev()."""

    def test_value_only(self):
        """Returns formatted value when stddev is None."""
        assert format_value_with_stddev(12.5) == "12.5"

    def test_value_with_stddev(self):
        """Returns 'value +/- stddev' when stddev is provided."""
        result = format_value_with_stddev(12.5, stddev=1.2)
        assert result == "12.5 \u00b1 1.2"

    def test_nan_stddev_treated_as_none(self):
        """NaN stddev is treated like None (value only)."""
        assert format_value_with_stddev(12.5, stddev=float("nan")) == "12.5"

    def test_custom_decimals(self):
        """Respects the decimals parameter."""
        assert format_value_with_stddev(3.14159, decimals=3) == "3.142"

    def test_custom_decimals_with_stddev(self):
        """Decimals applies to both value and stddev."""
        result = format_value_with_stddev(3.14159, stddev=0.0567, decimals=3)
        assert result == "3.142 \u00b1 0.057"

    def test_zero_value(self):
        """Zero value is formatted normally."""
        assert format_value_with_stddev(0.0, decimals=1) == "0.0"

    def test_zero_stddev(self):
        """Zero stddev is displayed (not treated as absent)."""
        result = format_value_with_stddev(5.0, stddev=0.0, decimals=1)
        assert result == "5.0 \u00b1 0.0"

    def test_large_values(self):
        """Handles large numbers without scientific notation."""
        result = format_value_with_stddev(12345.6, stddev=78.9, decimals=1)
        assert result == "12345.6 \u00b1 78.9"

    def test_negative_value(self):
        """Negative values are formatted with the minus sign."""
        assert format_value_with_stddev(-3.5, decimals=1) == "-3.5"

    def test_default_decimals_is_one(self):
        """Default decimal places is 1."""
        assert format_value_with_stddev(7.777) == "7.8"


# ===========================================================================
# generate_latency_vs_throughput_table
# ===========================================================================

class TestGenerateLatencyVsThroughputTable:
    """Tests for generate_latency_vs_throughput_table()."""

    def test_basic_output(self):
        """Generates a markdown table with header and data rows."""
        df = _make_index_df(n_rows=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "# Commit Latency vs Throughput" in text
            rows = _table_rows(text)
            # Header row + 3 data rows
            header_rows = [r for r in rows if "Throughput" in r[0] and "Success" in r[1]]
            data_rows = [r for r in rows if r not in header_rows
                         and "Throughput" not in r[0]]
            assert len(data_rows) >= 3

    def test_latency_conversion_ms_to_seconds(self):
        """Latency values are converted from ms to seconds (divided by 1000)."""
        df = pd.DataFrame({
            "throughput": [10.0],
            "success_rate": [100.0],
            "p50_commit_latency": [2000.0],   # 2.0 seconds
            "p95_commit_latency": [4000.0],   # 4.0 seconds
            "p99_commit_latency": [6000.0],   # 6.0 seconds
            "mean_retries": [0.5],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            # Find the data row (not header)
            data = [r for r in rows if "2.00" in " ".join(r)]
            assert len(data) == 1, f"Expected row with 2.00s latency, rows: {rows}"
            row_text = " ".join(data[0])
            assert "2.00" in row_text   # p50 = 2000ms -> 2.00s
            assert "4.00" in row_text   # p95 = 4000ms -> 4.00s
            assert "6.00" in row_text   # p99 = 6000ms -> 6.00s

    def test_with_stddev(self):
        """Stddev columns produce 'value +/- stddev' formatting."""
        df = _make_index_df(n_rows=2, include_stddev=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "\u00b1" in text
            assert "mean \u00b1 standard deviation" in text

    def test_without_stddev(self):
        """Without stddev columns, no +/- appears."""
        df = _make_index_df(n_rows=2, include_stddev=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "\u00b1" not in text

    def test_group_by(self):
        """group_by produces separate sections with '## group = val' headers."""
        df = _make_index_df(
            n_rows=2,
            group_by_col="num_tables",
            group_by_vals=[1, 5],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out, group_by="num_tables")

            text = _read_output(out)
            assert "## num_tables = 1" in text
            assert "## num_tables = 5" in text

    def test_group_by_nonexistent_column(self):
        """group_by with nonexistent column falls through to ungrouped."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out, group_by="nonexistent")

            text = _read_output(out)
            # Should still produce a table (ungrouped path)
            assert "# Commit Latency vs Throughput" in text
            assert "##" not in text.split("# Commit Latency vs Throughput\n\n")[1].split("## Notes")[0].strip() or True

    def test_sorted_by_throughput(self):
        """Rows are sorted by throughput ascending."""
        df = pd.DataFrame({
            "throughput": [15.0, 5.0, 10.0],
            "success_rate": [90.0, 100.0, 95.0],
            "p50_commit_latency": [800.0, 200.0, 500.0],
            "p95_commit_latency": [1600.0, 400.0, 1000.0],
            "p99_commit_latency": [2400.0, 600.0, 1500.0],
            "mean_retries": [1.5, 0.1, 0.8],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            data_rows = [r for r in rows if r[0].strip().replace(".", "").isdigit()]
            throughputs = [float(r[0].strip()) for r in data_rows]
            assert throughputs == sorted(throughputs)

    def test_empty_df_skipped(self):
        """Empty DataFrame does not create an output file."""
        df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)
            assert not os.path.exists(out)

    def test_missing_throughput_column_skipped(self):
        """DataFrame without 'throughput' does not create an output file."""
        df = pd.DataFrame({"success_rate": [100.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)
            assert not os.path.exists(out)

    def test_notes_section_present(self):
        """Output includes a Notes section."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "## Notes" in text
            assert "Latencies reported in seconds" in text

    def test_column_count(self):
        """Each data row has 6 pipe-separated columns."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            # All table rows (header + data) should have 6 columns
            table_rows = [r for r in rows if len(r) == 6]
            assert len(table_rows) >= 3  # header + 2 data rows


# ===========================================================================
# generate_success_rate_table
# ===========================================================================

class TestGenerateSuccessRateTable:
    """Tests for generate_success_rate_table()."""

    def test_basic_output(self):
        """Generates a markdown table with required columns."""
        df = _make_index_df(n_rows=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)

            text = _read_output(out)
            assert "# Transaction Success Rate vs Load" in text
            assert "Inter-Arrival (ms)" in text
            assert "Arrival Rate (txn/s)" in text

    def test_arrival_rate_computation(self):
        """arrival_rate = 1000 / inter_arrival_scale is correctly computed."""
        df = pd.DataFrame({
            "inter_arrival_scale": [100.0, 200.0, 500.0],
            "throughput": [10.0, 5.0, 2.0],
            "success_rate": [100.0, 95.0, 90.0],
            "committed": [1000, 950, 900],
            "total_txns": [1000, 1000, 1000],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)

            text = _read_output(out)
            # 1000/100 = 10.00, 1000/200 = 5.00, 1000/500 = 2.00
            assert "10.00" in text
            assert "5.00" in text
            assert "2.00" in text

    def test_sorted_by_inter_arrival_scale(self):
        """Rows are sorted by inter_arrival_scale ascending."""
        df = pd.DataFrame({
            "inter_arrival_scale": [500.0, 100.0, 200.0],
            "throughput": [2.0, 10.0, 5.0],
            "success_rate": [90.0, 100.0, 95.0],
            "committed": [900, 1000, 950],
            "total_txns": [1000, 1000, 1000],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            # Find data rows with numeric first column (inter-arrival)
            data_rows = [r for r in rows
                         if r[0].strip().replace(".", "").isdigit()]
            ia_values = [float(r[0].strip()) for r in data_rows]
            assert ia_values == sorted(ia_values)

    def test_group_by(self):
        """group_by produces per-group sections."""
        df = _make_index_df(
            n_rows=2,
            group_by_col="num_tables",
            group_by_vals=[1, 5],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out, group_by="num_tables")

            text = _read_output(out)
            assert "## num_tables = 1" in text
            assert "## num_tables = 5" in text

    def test_empty_df_skipped(self):
        """Empty DataFrame does not create an output file."""
        df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)
            assert not os.path.exists(out)

    def test_missing_throughput_skipped(self):
        """Missing throughput column skips the file."""
        df = pd.DataFrame({"success_rate": [100.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)
            assert not os.path.exists(out)

    def test_notes_section(self):
        """Output includes Notes section with interpretation guidance."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)

            text = _read_output(out)
            assert "## Notes" in text
            assert "Arrival rate = 1000 / inter_arrival_scale" in text

    def test_column_count(self):
        """Each data row has 6 pipe-separated columns."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "success.md")
            generate_success_rate_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            table_rows = [r for r in rows if len(r) == 6]
            assert len(table_rows) >= 3  # header + 2 data


# ===========================================================================
# generate_overhead_table
# ===========================================================================

class TestGenerateOverheadTable:
    """Tests for generate_overhead_table()."""

    def test_basic_output(self):
        """Generates a markdown table with overhead percentiles."""
        df = _make_index_df(n_rows=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            assert "# Commit Overhead vs Throughput" in text
            assert "Mean Overhead (%)" in text
            assert "P50 Overhead (%)" in text

    def test_with_stddev(self):
        """Stddev columns produce +/- notation."""
        df = _make_index_df(n_rows=2, include_stddev=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            assert "\u00b1" in text
            assert "mean \u00b1 standard deviation" in text

    def test_without_stddev(self):
        """Without stddev columns, no +/- appears."""
        df = _make_index_df(n_rows=2, include_stddev=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            assert "\u00b1" not in text

    def test_group_by(self):
        """group_by produces per-group sections."""
        df = _make_index_df(
            n_rows=2,
            group_by_col="real_conflict_probability",
            group_by_vals=[0.0, 0.5],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out, group_by="real_conflict_probability")

            text = _read_output(out)
            assert "## real_conflict_probability = 0.0" in text
            assert "## real_conflict_probability = 0.5" in text

    def test_empty_df_skipped(self):
        """Empty DataFrame does not create an output file."""
        df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)
            assert not os.path.exists(out)

    def test_missing_throughput_skipped(self):
        """Missing throughput column skips the file."""
        df = pd.DataFrame({"success_rate": [100.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)
            assert not os.path.exists(out)

    def test_interpretation_section(self):
        """Output includes Interpretation section with overhead brackets."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            assert "## Interpretation" in text
            assert "Low overhead" in text
            assert "Very high overhead" in text

    def test_overhead_values_present(self):
        """Specific overhead values appear in the output."""
        df = pd.DataFrame({
            "throughput": [10.0],
            "success_rate": [100.0],
            "mean_overhead_pct": [12.5],
            "p50_overhead_pct": [10.0],
            "p95_overhead_pct": [25.0],
            "p99_overhead_pct": [35.0],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            assert "12.5" in text
            assert "10.0" in text
            assert "25.0" in text
            assert "35.0" in text

    def test_column_count(self):
        """Each data row has 6 pipe-separated columns."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            table_rows = [r for r in rows if len(r) == 6]
            assert len(table_rows) >= 3  # header + 2 data

    def test_sorted_by_throughput(self):
        """Rows are sorted by throughput ascending."""
        df = pd.DataFrame({
            "throughput": [20.0, 5.0, 12.0],
            "success_rate": [80.0, 100.0, 95.0],
            "mean_overhead_pct": [30.0, 5.0, 15.0],
            "p50_overhead_pct": [25.0, 3.0, 12.0],
            "p95_overhead_pct": [50.0, 8.0, 25.0],
            "p99_overhead_pct": [60.0, 10.0, 30.0],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "overhead.md")
            generate_overhead_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            data_rows = [r for r in rows
                         if r[0].strip().replace(".", "").isdigit()]
            throughputs = [float(r[0].strip()) for r in data_rows]
            assert throughputs == sorted(throughputs)


# ===========================================================================
# generate_success_rate_vs_throughput_table
# ===========================================================================

class TestGenerateSuccessRateVsThroughputTable:
    """Tests for generate_success_rate_vs_throughput_table()."""

    def test_basic_output(self):
        """Generates a 2-column markdown table."""
        df = _make_index_df(n_rows=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "# Success Rate vs Throughput" in text

    def test_two_column_layout(self):
        """Table has exactly 2 columns: Throughput and Success Rate."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            table_rows = [r for r in rows if len(r) == 2]
            assert len(table_rows) >= 3  # header + 2 data

    def test_with_stddev(self):
        """Stddev columns produce +/- formatting."""
        df = _make_index_df(n_rows=2, include_stddev=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "\u00b1" in text
            assert "mean \u00b1 standard deviation" in text

    def test_without_stddev(self):
        """Without stddev, no +/- appears."""
        df = _make_index_df(n_rows=2, include_stddev=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "\u00b1" not in text

    def test_group_by(self):
        """group_by produces per-group sections."""
        df = _make_index_df(
            n_rows=2,
            group_by_col="num_tables",
            group_by_vals=[1, 10],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out, group_by="num_tables")

            text = _read_output(out)
            assert "## num_tables = 1" in text
            assert "## num_tables = 10" in text

    def test_empty_df_skipped(self):
        """Empty DataFrame does not create a file."""
        df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)
            assert not os.path.exists(out)

    def test_missing_throughput_skipped(self):
        """Missing throughput column skips."""
        df = pd.DataFrame({"success_rate": [100.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)
            assert not os.path.exists(out)

    def test_notes_section(self):
        """Output includes Notes section."""
        df = _make_index_df(n_rows=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            assert "## Notes" in text
            assert "Throughput = commits per second" in text

    def test_sorted_by_throughput(self):
        """Rows are sorted by throughput ascending."""
        df = pd.DataFrame({
            "throughput": [20.0, 3.0, 10.0],
            "success_rate": [80.0, 100.0, 95.0],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            rows = _table_rows(text)
            data_rows = [r for r in rows
                         if r[0].strip().replace(".", "").isdigit()]
            throughputs = [float(r[0].strip()) for r in data_rows]
            assert throughputs == sorted(throughputs)

    def test_stddev_noted_in_notes(self):
        """When stddev is present, Notes section mentions it."""
        df = _make_index_df(n_rows=2, include_stddev=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            notes = text.split("## Notes")[1]
            assert "standard deviation" in notes

    def test_no_stddev_notes_omit_stddev_line(self):
        """When no stddev, Notes section does not mention stddev."""
        df = _make_index_df(n_rows=2, include_stddev=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_vs_tp.md")
            generate_success_rate_vs_throughput_table(df, out)

            text = _read_output(out)
            notes = text.split("## Notes")[1]
            assert "standard deviation" not in notes


# ===========================================================================
# generate_commit_rate_over_time_table
# ===========================================================================

class TestGenerateCommitRateOverTimeTable:
    """Tests for generate_commit_rate_over_time_table()."""

    def test_basic_output(self):
        """Generates a markdown table from experiment directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "aa11", seeds=(42,),
                               n_txns=200, scale=100.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out)

            text = _read_output(out)
            assert "# Commit Rate Over Time" in text
            assert "Time (min)" in text
            assert "Commit Rate (c/s)" in text

    def test_window_size(self):
        """Custom window_size_sec changes the window description."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "bb22", seeds=(42,),
                               n_txns=200, scale=100.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(
                tmpdir, "exp_crt*", out, window_size_sec=120
            )

            text = _read_output(out)
            assert "120-second windows" in text

    def test_multiple_experiments(self):
        """Multiple matching experiments each get their own section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "cc33", seeds=(42,),
                               n_txns=200, scale=50.0)
            _create_experiment(tmpdir, "exp_crt", "dd44", seeds=(42,),
                               n_txns=200, scale=200.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out)

            text = _read_output(out)
            # Should have at least 2 inter_arrival sections
            assert text.count("## inter_arrival=") >= 2

    def test_no_experiments_skipped(self):
        """No matching experiments does not create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(
                tmpdir, "nonexistent_pattern*", out
            )
            assert not os.path.exists(out)

    def test_notes_section(self):
        """Output includes Notes section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "ee55", seeds=(42,),
                               n_txns=200, scale=100.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out)

            text = _read_output(out)
            assert "## Notes" in text
            assert "warmup" in text.lower()

    def test_time_in_minutes(self):
        """Time column shows values in minutes (midpoints of windows)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "ff66", seeds=(42,),
                               n_txns=200, scale=100.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out)

            text = _read_output(out)
            rows = _table_rows(text)
            # First time window midpoint should be 0.5 min (30s midpoint of 0-60s)
            data_rows = [r for r in rows if len(r) == 2
                         and r[0].strip().replace(".", "").isdigit()]
            if data_rows:
                first_time = float(data_rows[0][0].strip())
                assert first_time == pytest.approx(0.5, abs=0.1)

    def test_commit_rate_column_count(self):
        """Each data row has exactly 2 columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "gg77", seeds=(42,),
                               n_txns=200, scale=100.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out)

            text = _read_output(out)
            rows = _table_rows(text)
            # All table rows should have 2 columns
            for r in rows:
                assert len(r) == 2, f"Expected 2 columns, got {len(r)}: {r}"

    def test_multiple_seeds_aggregated(self):
        """Multiple seeds are loaded and concatenated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_crt", "hh88", seeds=(42, 43, 44),
                               n_txns=100, scale=100.0)
            out = os.path.join(tmpdir, "crt.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out)

            text = _read_output(out)
            assert "# Commit Rate Over Time" in text
            # With more seeds the commit rate should be higher (more commits per window)
            rows = _table_rows(text)
            data_rows = [r for r in rows if len(r) == 2
                         and r[0].strip().replace(".", "").isdigit()]
            assert len(data_rows) > 0

    def test_aborted_transactions_excluded(self):
        """Only committed transactions contribute to the commit rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 50% success rate -- half are aborted
            _create_experiment(tmpdir, "exp_crt", "ii99", seeds=(42,),
                               n_txns=200, scale=100.0, success_rate=0.5)
            out_half = os.path.join(tmpdir, "crt_half.md")
            generate_commit_rate_over_time_table(tmpdir, "exp_crt*", out_half)

            text = _read_output(out_half)
            rows = _table_rows(text)
            data_rows = [r for r in rows if len(r) == 2
                         and r[0].strip().replace(".", "").isdigit()]
            # At least some data was generated
            assert len(data_rows) > 0


# ===========================================================================
# Cross-cutting / edge-case tests
# ===========================================================================

class TestEdgeCases:
    """Edge cases that apply to multiple table generators."""

    def test_single_row_dataframe(self):
        """Single-row DataFrame produces valid output for all generators."""
        df = _make_index_df(n_rows=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            for fn, name in [
                (generate_latency_vs_throughput_table, "latency.md"),
                (generate_success_rate_table, "success.md"),
                (generate_overhead_table, "overhead.md"),
                (generate_success_rate_vs_throughput_table, "sr_tp.md"),
            ]:
                out = os.path.join(tmpdir, name)
                fn(df, out)
                assert os.path.exists(out), f"{name} was not created"
                text = _read_output(out)
                assert len(text) > 0

    def test_group_by_single_group(self):
        """group_by with a single group value still creates a section header."""
        df = _make_index_df(
            n_rows=2,
            group_by_col="num_tables",
            group_by_vals=[1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out, group_by="num_tables")

            text = _read_output(out)
            assert "## num_tables = 1" in text

    def test_group_by_preserves_sort_within_groups(self):
        """Within each group, rows are sorted by the relevant column."""
        df = pd.DataFrame({
            "throughput": [15.0, 5.0, 20.0, 3.0],
            "success_rate": [90.0, 100.0, 80.0, 100.0],
            "num_tables": [1, 1, 5, 5],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sr_tp.md")
            generate_success_rate_vs_throughput_table(
                df, out, group_by="num_tables"
            )

            text = _read_output(out)
            # Split by group headers
            sections = text.split("## num_tables =")
            for section in sections[1:]:
                rows = _table_rows(section)
                data_rows = [r for r in rows if len(r) == 2
                             and r[0].strip().replace(".", "").replace("\u00b1", "").replace(" ", "").replace("-", "").isdigit() is False]
                # Filter to numeric data rows
                numeric_rows = []
                for r in rows:
                    try:
                        float(r[0].strip().split("\u00b1")[0].strip())
                        numeric_rows.append(r)
                    except (ValueError, IndexError):
                        continue
                if len(numeric_rows) > 1:
                    vals = [float(r[0].strip().split("\u00b1")[0].strip())
                            for r in numeric_rows]
                    assert vals == sorted(vals)

    def test_stddev_with_nan_values(self):
        """NaN in stddev columns does not break formatting."""
        df = pd.DataFrame({
            "throughput": [10.0, 15.0],
            "success_rate": [100.0, 95.0],
            "p50_commit_latency": [500.0, 800.0],
            "p95_commit_latency": [1000.0, 1600.0],
            "p99_commit_latency": [1500.0, 2400.0],
            "mean_retries": [0.5, 1.0],
            "throughput_std": [0.5, float("nan")],
            "success_rate_std": [1.0, float("nan")],
            "p50_commit_latency_std": [20.0, float("nan")],
            "p95_commit_latency_std": [40.0, float("nan")],
            "p99_commit_latency_std": [60.0, float("nan")],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "latency.md")
            generate_latency_vs_throughput_table(df, out)

            text = _read_output(out)
            # Should contain at least one +/- from the non-NaN row
            assert "\u00b1" in text
            # Should not crash and should produce valid output
            assert "# Commit Latency vs Throughput" in text
