#!/usr/bin/env python3
"""Dump results.parquet file contents to stdout.

This tool displays the schema and contents of results.parquet files generated
by the Icecap simulator. Supports various output formats and filtering options.

Supports both:
- Individual results.parquet files (small, < 10 MiB)
- Consolidated parquet files (large, 4+ GiB) with streaming

Schema:
    txn_id              int64       Transaction ID
    t_submit            int64       Submission time (ms from simulation start)
    t_runtime           int64       Transaction runtime (ms)
    t_commit            int64       Commit completion time (ms from simulation start)
    commit_latency      int64       Time spent in commit protocol (ms)
    total_latency       int64       Total transaction latency (ms)
    n_retries           int8        Number of commit retries
    n_tables_read       int8        Number of tables read
    n_tables_written    int8        Number of tables written
    status              object      Transaction status ('committed' or 'aborted')

Consolidated schema (additional columns):
    exp_name            string      Experiment name
    exp_hash            string      Experiment hash
    seed                int64       Random seed
    config              map         Configuration parameters

Note: All time/latency values are integers (ms). Using int64 eliminates
floating point inaccuracy and reduces file size by 24% vs float64.

Usage:
    # Show schema and summary
    python scripts/dump_results.py results.parquet

    # Show all rows (small files only!)
    python scripts/dump_results.py results.parquet --all

    # Show first 20 rows
    python scripts/dump_results.py results.parquet --head 20

    # Filter consolidated file by experiment
    python scripts/dump_results.py consolidated.parquet --exp-name exp2_1_single_table_false --exp-hash 0be55863 --seed 2953848217

    # Show committed transactions only
    python scripts/dump_results.py results.parquet --status committed

    # Output as CSV
    python scripts/dump_results.py results.parquet --format csv

    # Show statistics by status
    python scripts/dump_results.py results.parquet --stats
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


def format_ms(ms):
    """Format milliseconds as human-readable time."""
    if pd.isna(ms):
        return 'N/A'
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_schema(df):
    """Print schema information."""
    print("=" * 80)
    print("SCHEMA")
    print("=" * 80)
    print()
    print(f"{'Column':<20} {'Type':<15} {'Description'}")
    print("-" * 80)

    descriptions = {
        'txn_id': 'Transaction ID',
        't_submit': 'Submission time (ms from sim start)',
        't_runtime': 'Transaction runtime (ms)',
        't_commit': 'Commit completion time (ms)',
        'commit_latency': 'Time in commit protocol (ms)',
        'total_latency': 'Total transaction latency (ms)',
        'n_retries': 'Number of commit retries',
        'n_tables_read': 'Number of tables read',
        'n_tables_written': 'Number of tables written',
        'status': 'Status (committed/aborted)'
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        desc = descriptions.get(col, '')
        print(f"{col:<20} {dtype:<15} {desc}")
    print()


def print_summary(df):
    """Print summary statistics."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total rows:        {len(df):,}")
    print(f"Committed:         {len(df[df['status']=='committed']):,} ({100*len(df[df['status']=='committed'])/len(df):.1f}%)")
    print(f"Aborted:           {len(df[df['status']=='aborted']):,} ({100*len(df[df['status']=='aborted'])/len(df):.1f}%)")
    print()

    print("Time Range:")
    print(f"  First submission:  {format_ms(df['t_submit'].min())} ({df['t_submit'].min()}ms)")
    print(f"  Last submission:   {format_ms(df['t_submit'].max())} ({df['t_submit'].max()}ms)")
    print(f"  Simulation span:   {format_ms(df['t_submit'].max() - df['t_submit'].min())}")
    print()

    committed = df[df['status'] == 'committed']
    if len(committed) > 0:
        print("Committed Transactions:")
        print(f"  Mean commit latency:  {format_ms(committed['commit_latency'].mean())} ({committed['commit_latency'].mean():.1f}ms)")
        print(f"  P50 commit latency:   {format_ms(committed['commit_latency'].quantile(0.5))} ({committed['commit_latency'].quantile(0.5):.1f}ms)")
        print(f"  P95 commit latency:   {format_ms(committed['commit_latency'].quantile(0.95))} ({committed['commit_latency'].quantile(0.95):.1f}ms)")
        print(f"  P99 commit latency:   {format_ms(committed['commit_latency'].quantile(0.99))} ({committed['commit_latency'].quantile(0.99):.1f}ms)")
        print(f"  Mean retries:         {committed['n_retries'].mean():.2f}")
        print(f"  Max retries:          {committed['n_retries'].max()}")
        print()


def print_statistics(df):
    """Print detailed statistics by status."""
    print("=" * 80)
    print("STATISTICS BY STATUS")
    print("=" * 80)
    print()

    for status in df['status'].unique():
        subset = df[df['status'] == status]
        print(f"{status.upper()} ({len(subset)} transactions, {100*len(subset)/len(df):.1f}%)")
        print("-" * 40)

        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'txn_id':
                continue
            print(f"  {col}:")
            print(f"    mean:  {subset[col].mean():.2f}")
            print(f"    min:   {subset[col].min():.2f}")
            print(f"    p50:   {subset[col].quantile(0.5):.2f}")
            print(f"    p95:   {subset[col].quantile(0.95):.2f}")
            print(f"    max:   {subset[col].max():.2f}")
        print()


def print_data(df, format='table', max_rows=None):
    """Print data in specified format."""
    if max_rows:
        df = df.head(max_rows)

    if format == 'csv':
        print(df.to_csv(index=False))
    elif format == 'json':
        print(df.to_json(orient='records', indent=2))
    elif format == 'table':
        # Pretty print as table
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', None,
                               'display.max_colwidth', 20):
            print(df.to_string())
    else:
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Dump results.parquet file contents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show schema and summary
  %(prog)s results.parquet

  # Show all transactions
  %(prog)s results.parquet --all

  # Show first 50 rows
  %(prog)s results.parquet --head 50

  # Show committed transactions only
  %(prog)s results.parquet --status committed

  # Show transactions submitted in first 10 minutes
  %(prog)s results.parquet --time-range 0 600000

  # Export as CSV
  %(prog)s results.parquet --all --format csv > output.csv

  # Show detailed statistics
  %(prog)s results.parquet --stats
        """
    )

    parser.add_argument('file', type=str, help='Path to results.parquet file')
    parser.add_argument('--schema', action='store_true', help='Show schema only')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    parser.add_argument('--all', action='store_true', help='Show all rows (use with caution on large files)')
    parser.add_argument('--head', type=int, metavar='N', help='Show first N rows')
    parser.add_argument('--tail', type=int, metavar='N', help='Show last N rows')

    # Consolidated file filtering
    parser.add_argument('--exp-name', type=str, help='Filter by experiment name (consolidated files)')
    parser.add_argument('--exp-hash', type=str, help='Filter by experiment hash (consolidated files)')
    parser.add_argument('--seed', type=int, help='Filter by seed (consolidated files)')

    # Data filtering
    parser.add_argument('--status', choices=['committed', 'aborted'], help='Filter by status')
    parser.add_argument('--time-range', nargs=2, type=int, metavar=('START', 'END'),
                        help='Filter by submission time range (ms)')
    parser.add_argument('--retries-min', type=int, metavar='N',
                        help='Filter transactions with at least N retries')

    # Output formatting
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--columns', nargs='+', metavar='COL',
                        help='Show only specified columns')
    parser.add_argument('--sort', metavar='COLUMN', help='Sort by column')
    parser.add_argument('--no-schema', action='store_true', help='Skip schema display')
    parser.add_argument('--no-summary', action='store_true', help='Skip summary display')

    args = parser.parse_args()

    # Load data
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Check file size to determine loading strategy
        file_size_mb = file_path.stat().st_size / (1024**2)

        # Build filters for consolidated files
        filters = []
        if args.exp_name:
            filters.append(('exp_name', '==', args.exp_name))
        if args.exp_hash:
            filters.append(('exp_hash', '==', args.exp_hash))
        if args.seed:
            filters.append(('seed', '==', args.seed))

        # Use predicate pushdown for large files or when filters are specified
        if file_size_mb > 100 or filters:
            if file_size_mb > 100:
                print(f"Large file detected ({file_size_mb:.1f} MB) - using predicate pushdown", file=sys.stderr)
            if filters:
                print(f"Applying filters: {filters}", file=sys.stderr)

            df = pd.read_parquet(file_path, filters=filters if filters else None)

            # Drop consolidated metadata columns if present
            metadata_cols = ['exp_name', 'exp_hash', 'seed', 'config']
            cols_to_drop = [c for c in metadata_cols if c in df.columns]
            if cols_to_drop and not args.all and not args.head:
                df = df.drop(columns=cols_to_drop)
        else:
            # Small file - load normally
            df = pd.read_parquet(file_path)

    except Exception as e:
        print(f"Error reading parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    if len(df) == 0:
        print("No data found matching the specified filters", file=sys.stderr)
        sys.exit(0)

    # Apply filters
    if args.status:
        df = df[df['status'] == args.status]

    if args.time_range:
        start, end = args.time_range
        df = df[(df['t_submit'] >= start) & (df['t_submit'] < end)]

    if args.retries_min is not None:
        df = df[df['n_retries'] >= args.retries_min]

    # Apply column selection
    if args.columns:
        missing = set(args.columns) - set(df.columns)
        if missing:
            print(f"Error: Unknown columns: {missing}", file=sys.stderr)
            sys.exit(1)
        df = df[args.columns]

    # Apply sorting
    if args.sort:
        if args.sort not in df.columns:
            print(f"Error: Unknown column: {args.sort}", file=sys.stderr)
            sys.exit(1)
        df = df.sort_values(args.sort)

    # Show schema
    if args.schema or (not args.no_schema and not args.all and not args.head and not args.tail):
        print_schema(df)

    # Show summary
    if args.summary or (not args.no_summary and not args.all and not args.head and not args.tail):
        print_summary(df)

    # Show statistics
    if args.stats:
        print_statistics(df)

    # Show data
    if args.all:
        print("=" * 80)
        print(f"DATA ({len(df)} rows)")
        print("=" * 80)
        print()
        print_data(df, format=args.format)
    elif args.head:
        print("=" * 80)
        print(f"FIRST {args.head} ROWS")
        print("=" * 80)
        print()
        print_data(df.head(args.head), format=args.format)
    elif args.tail:
        print("=" * 80)
        print(f"LAST {args.tail} ROWS")
        print("=" * 80)
        print()
        print_data(df.tail(args.tail), format=args.format)


if __name__ == '__main__':
    main()
