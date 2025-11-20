import math
import sys

import numpy as np
import pandas as pd
from hdrh.histogram import HdrHistogram

def truncated_zipf_pmf(n, s):
    """
    Compute the truncated Zipf PMF for ranks 1 through n with exponent s.
    Returns a probability vector of length n.
    (OpenAI ChatGPT o1)
    """
    ranks = np.arange(1, n+1)
    weights = 1.0 / (ranks ** s)
    pmf = weights / weights.sum()
    return pmf

def lognormal_params_from_mean_and_sigma(mean_runtime_ms: float, sigma: float) -> (float, float):
    """
    Given the desired average (mean) runtime of a lognormal distribution and
    the chosen sigma (std. dev.) of the underlying normal distribution,
    compute the mu parameter for the underlying normal distribution.

    Parameters
    ----------
    mean_runtime_ms : float
        The desired mean of the lognormal distribution (e.g., 10,000 ms for 10 seconds).
    sigma : float
        The standard deviation of the underlying normal distribution that
        generates the lognormal distribution.

    Returns
    -------
    mu : float
        The mu parameter of the underlying normal distribution.
    sigma : float
        The sigma parameter of the underlying normal distribution (unchanged).
    """
    mu = math.log(mean_runtime_ms) - (sigma ** 2 / 2.0)
    return mu, sigma

class Stats:
    def __init__(self):
        # TODO compute max w.r.t. longest total time w/ retries
        self.latency = HdrHistogram(1, 10 * 60 * 1000, 3)
        self.txn_total = 0
        self.txn_committed = 0
        self.txn_aborted = 0

        # Collect detailed transaction data for export
        self.transactions = []

    def commit(self, txn):
        self.txn_total += 1
        self.txn_committed += 1
        commit_latency = txn.t_commit - txn.t_runtime - txn.t_submit
        self.latency.record_value(commit_latency)

        # Record transaction details
        self.transactions.append({
            'txn_id': txn.id,
            't_submit': txn.t_submit,
            't_runtime': txn.t_runtime,
            't_commit': txn.t_commit,
            'commit_latency': commit_latency,
            'total_latency': txn.t_commit - txn.t_submit,
            'n_retries': txn.n_retries,
            'n_tables_read': len(txn.v_tblr),
            'n_tables_written': len(txn.v_tblw),
            'status': 'committed'
        })

    def abort(self, txn):
        self.txn_total += 1
        self.txn_aborted += 1

        # Record aborted transaction details
        self.transactions.append({
            'txn_id': txn.id,
            't_submit': txn.t_submit,
            't_runtime': txn.t_runtime,
            't_commit': -1,
            'commit_latency': -1,
            'total_latency': txn.t_abort - txn.t_submit,
            'n_retries': txn.n_retries,
            'n_tables_read': len(txn.v_tblr),
            'n_tables_written': len(txn.v_tblw),
            'status': 'aborted'
        })

    def print_summary(self, out=sys.stdout):
        assert self.txn_committed + self.txn_aborted == self.txn_total
        self.latency.output_percentile_distribution(out_file=out)

    def export_parquet(self, output_path: str):
        """Export transaction data to Parquet format."""
        df = pd.DataFrame(self.transactions)
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)

        # Also print summary statistics
        print(f"\nSimulation Summary:")
        print(f"  Total transactions: {self.txn_total}")
        print(f"  Committed: {self.txn_committed} ({100*self.txn_committed/self.txn_total:.1f}%)")
        print(f"  Aborted: {self.txn_aborted} ({100*self.txn_aborted/self.txn_total:.1f}%)")
        if self.txn_committed > 0:
            committed_df = df[df['status'] == 'committed']
            print(f"  Commit latency (ms):")
            print(f"    Mean: {committed_df['commit_latency'].mean():.2f}")
            print(f"    Median: {committed_df['commit_latency'].median():.2f}")
            print(f"    P95: {committed_df['commit_latency'].quantile(0.95):.2f}")
            print(f"    P99: {committed_df['commit_latency'].quantile(0.99):.2f}")
            print(f"  Retries per transaction:")
            print(f"    Mean: {committed_df['n_retries'].mean():.2f}")
            print(f"    Max: {committed_df['n_retries'].max()}")