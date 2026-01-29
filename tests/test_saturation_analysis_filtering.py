#!/usr/bin/env python
"""
Tests for saturation_analysis filtering functionality.

Tests the parse_filter_expression() and apply_filters() functions.
"""

import pytest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from endive.saturation_analysis import parse_filter_expression, apply_filters


class TestParseFilterExpression:
    """Test filter expression parsing."""

    def test_parse_equals_int(self):
        """Test parsing == operator with integer value."""
        param, op, value = parse_filter_expression("t_cas_mean==50")
        assert param == "t_cas_mean"
        assert op == "=="
        assert value == 50
        assert isinstance(value, int)

    def test_parse_equals_float(self):
        """Test parsing == operator with float value."""
        param, op, value = parse_filter_expression("threshold==0.5")
        assert param == "threshold"
        assert op == "=="
        assert value == 0.5
        assert isinstance(value, float)

    def test_parse_equals_string(self):
        """Test parsing == operator with string value."""
        param, op, value = parse_filter_expression('label=="exp5_1"')
        assert param == "label"
        assert op == "=="
        assert value == "exp5_1"
        assert isinstance(value, str)

    def test_parse_not_equals(self):
        """Test parsing != operator."""
        param, op, value = parse_filter_expression("num_tables!=1")
        assert param == "num_tables"
        assert op == "!="
        assert value == 1

    def test_parse_less_than(self):
        """Test parsing < operator."""
        param, op, value = parse_filter_expression("latency<100")
        assert param == "latency"
        assert op == "<"
        assert value == 100

    def test_parse_less_than_or_equal(self):
        """Test parsing <= operator."""
        param, op, value = parse_filter_expression("num_groups<=5")
        assert param == "num_groups"
        assert op == "<="
        assert value == 5

    def test_parse_greater_than(self):
        """Test parsing > operator."""
        param, op, value = parse_filter_expression("throughput>100")
        assert param == "throughput"
        assert op == ">"
        assert value == 100

    def test_parse_greater_than_or_equal(self):
        """Test parsing >= operator."""
        param, op, value = parse_filter_expression("t_cas_mean>=50")
        assert param == "t_cas_mean"
        assert op == ">="
        assert value == 50

    def test_parse_with_whitespace(self):
        """Test parsing with whitespace around operator."""
        param, op, value = parse_filter_expression("  t_cas_mean  ==  50  ")
        assert param == "t_cas_mean"
        assert op == "=="
        assert value == 50

    def test_parse_invalid_no_operator(self):
        """Test parsing invalid expression with no operator."""
        with pytest.raises(ValueError, match="No valid operator"):
            parse_filter_expression("t_cas_mean50")

    def test_parse_invalid_empty_parts(self):
        """Test parsing invalid expression with empty parts."""
        with pytest.raises(ValueError, match="Invalid filter expression"):
            parse_filter_expression("==50")

    def test_operator_precedence(self):
        """Test that >= is matched before > (operator precedence)."""
        param, op, value = parse_filter_expression("x>=5")
        assert op == ">="
        assert value == 5


class TestApplyFilters:
    """Test filter application to DataFrames."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            't_cas_mean': [15, 50, 100, 200, 500, 1000],
            'num_tables': [1, 1, 5, 5, 10, 10],
            'num_groups': [1, 2, 5, 10, 20, 20],
            'throughput': [100, 80, 60, 40, 20, 10],
            'latency': [10, 20, 30, 50, 100, 200],
            'label': ['a', 'b', 'c', 'd', 'e', 'f']
        })

    def test_apply_single_equals_filter(self, sample_df):
        """Test applying single == filter."""
        result = apply_filters(sample_df, ["t_cas_mean==50"])
        assert len(result) == 1
        assert result.iloc[0]['t_cas_mean'] == 50

    def test_apply_single_greater_than_filter(self, sample_df):
        """Test applying single > filter."""
        result = apply_filters(sample_df, ["t_cas_mean>100"])
        assert len(result) == 3
        assert all(result['t_cas_mean'] > 100)

    def test_apply_single_less_than_or_equal_filter(self, sample_df):
        """Test applying single <= filter."""
        result = apply_filters(sample_df, ["num_tables<=5"])
        assert len(result) == 4
        assert all(result['num_tables'] <= 5)

    def test_apply_multiple_filters_and_logic(self, sample_df):
        """Test applying multiple filters with AND logic."""
        result = apply_filters(sample_df, ["t_cas_mean>=50", "num_tables<=5"])
        assert len(result) == 3  # Rows 1, 2, 3 match (t_cas_mean: 50, 100, 200)
        assert all(result['t_cas_mean'] >= 50)
        assert all(result['num_tables'] <= 5)

    def test_apply_filters_result_ordering(self, sample_df):
        """Test that filter preserves DataFrame ordering."""
        result = apply_filters(sample_df, ["num_groups>=5"])
        # Check that result is a subset with preserved order
        expected_indices = [2, 3, 4, 5]
        assert list(result.index) == expected_indices

    def test_apply_no_filters(self, sample_df):
        """Test that empty filter list returns original DataFrame."""
        result = apply_filters(sample_df, [])
        assert len(result) == len(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_apply_filter_no_matches(self, sample_df):
        """Test filter that matches no rows."""
        result = apply_filters(sample_df, ["t_cas_mean>2000"])
        assert len(result) == 0

    def test_apply_filter_all_matches(self, sample_df):
        """Test filter that matches all rows."""
        result = apply_filters(sample_df, ["t_cas_mean>=15"])
        assert len(result) == len(sample_df)

    def test_apply_filter_not_equals(self, sample_df):
        """Test != operator."""
        result = apply_filters(sample_df, ["num_tables!=1"])
        assert len(result) == 4
        assert all(result['num_tables'] != 1)

    def test_apply_filter_string_value(self, sample_df):
        """Test filtering on string column."""
        result = apply_filters(sample_df, ['label=="c"'])
        assert len(result) == 1
        assert result.iloc[0]['label'] == 'c'

    def test_apply_filter_invalid_column(self, sample_df):
        """Test error when filter references non-existent column."""
        with pytest.raises(ValueError, match="Filter parameter 'nonexistent' not found"):
            apply_filters(sample_df, ["nonexistent==50"])

    def test_apply_filter_creates_copy(self, sample_df):
        """Test that filtering returns a copy, not a view."""
        result = apply_filters(sample_df, ["t_cas_mean==50"])
        # Modify result
        result.loc[result.index[0], 't_cas_mean'] = 999
        # Original should be unchanged
        assert sample_df.iloc[1]['t_cas_mean'] == 50

    def test_apply_complex_multiple_filters(self, sample_df):
        """Test complex multi-filter scenario."""
        # Filter for experiments with:
        # - catalog latency between 50 and 500 (inclusive)
        # - num_tables >= 5
        # - throughput > 30
        result = apply_filters(sample_df, [
            "t_cas_mean>=50",
            "t_cas_mean<=500",
            "num_tables>=5",
            "throughput>30"
        ])
        assert len(result) == 2  # Should match rows with t_cas_mean=100,200
        assert all(result['t_cas_mean'] >= 50)
        assert all(result['t_cas_mean'] <= 500)
        assert all(result['num_tables'] >= 5)
        assert all(result['throughput'] > 30)

    def test_apply_filter_float_comparison(self, sample_df):
        """Test filtering with float values."""
        df_with_floats = sample_df.copy()
        df_with_floats['ratio'] = [0.1, 0.5, 0.7, 0.9, 1.2, 1.5]

        result = apply_filters(df_with_floats, ["ratio>=0.5", "ratio<1.0"])
        assert len(result) == 3
        assert all(result['ratio'] >= 0.5)
        assert all(result['ratio'] < 1.0)


class TestFilteringEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test filtering on empty DataFrame."""
        df = pd.DataFrame({'t_cas_mean': [], 'num_tables': []})
        result = apply_filters(df, ["t_cas_mean==50"])
        assert len(result) == 0

    def test_single_row_dataframe(self):
        """Test filtering on single-row DataFrame."""
        df = pd.DataFrame({'t_cas_mean': [50], 'num_tables': [1]})
        result = apply_filters(df, ["t_cas_mean==50"])
        assert len(result) == 1

    def test_filter_with_none_values(self):
        """Test filtering with None/NaN values in DataFrame."""
        df = pd.DataFrame({
            't_cas_mean': [15, None, 100, pd.NA],
            'num_tables': [1, 5, 10, 20]
        })
        # Filter should not match None/NA values
        result = apply_filters(df, ["t_cas_mean==15"])
        assert len(result) == 1
        assert result.iloc[0]['t_cas_mean'] == 15

    def test_multiple_filters_progressive_reduction(self):
        """Test that filters are applied progressively."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500]
        })

        # First filter: a >= 2 (keeps 4 rows)
        # Second filter: b < 40 (keeps 2 rows)
        # Third filter: c <= 300 (keeps 2 rows)
        result = apply_filters(df, ["a>=2", "b<40", "c<=300"])
        assert len(result) == 2
        assert list(result['a']) == [2, 3]


class TestFilteringIntegration:
    """Integration tests for filtering in realistic scenarios."""

    def test_exp5_2_filtering_scenario(self):
        """Test realistic exp5.2 filtering scenario."""
        # Simulate exp5.2 data: 6 T_CAS × 4 num_tables × 9 loads = 216 combinations
        data = []
        for t_cas in [15, 50, 100, 200, 500, 1000]:
            for num_tables in [1, 5, 20, 50]:
                for load in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
                    data.append({
                        't_cas_mean': t_cas,
                        'num_tables': num_tables,
                        'inter_arrival_scale': load,
                        'throughput': 1000 / (t_cas + load / 100),  # Dummy calculation
                        'latency': t_cas + load / 50  # Dummy calculation
                    })

        df = pd.DataFrame(data)
        assert len(df) == 216  # 6 × 4 × 9

        # Filter to single T_CAS value
        result = apply_filters(df, ["t_cas_mean==50"])
        assert len(result) == 36  # 4 tables × 9 loads
        assert all(result['t_cas_mean'] == 50)

        # Filter to T_CAS=50 and num_tables=5
        result = apply_filters(df, ["t_cas_mean==50", "num_tables==5"])
        assert len(result) == 9  # 9 loads
        assert all(result['t_cas_mean'] == 50)
        assert all(result['num_tables'] == 5)

    def test_filter_preserves_dataframe_dtypes(self):
        """Test that filtering preserves column data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })

        original_dtypes = df.dtypes.to_dict()
        result = apply_filters(df, ["int_col>=2"])

        # Check dtypes are preserved
        for col, dtype in original_dtypes.items():
            assert result[col].dtype == dtype


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
