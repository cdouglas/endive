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

        # Conflict resolution statistics (CAS mode)
        self.false_conflicts = 0  # Version changed, no data overlap
        self.real_conflicts = 0   # Overlapping data changes requiring manifest file operations
        self.manifest_files_read = 0
        self.manifest_files_written = 0

        # Append mode statistics
        self.append_physical_success = 0   # Append landed at expected offset
        self.append_logical_success = 0    # + verification passed -> commit
        self.append_logical_conflict = 0   # Append landed but conflict detected
        self.append_physical_failure = 0   # Offset moved, append failed
        self.append_compactions_triggered = 0  # Sealed transactions triggering compaction
        self.append_compactions_completed = 0  # Successful compaction CAS operations
        self.append_dedup_hits = 0         # Transaction already committed (deduplication)

        # Collect detailed transaction data for export
        self.transactions = []

    def commit(self, txn):
        self.txn_total += 1
        self.txn_committed += 1
        commit_latency = txn.t_commit - txn.t_runtime - txn.t_submit
        self.latency.record_value(commit_latency)

        # Record transaction details
        # Note: All time/latency fields are int64 (ms), not float64
        # This eliminates floating point inaccuracy and reduces file size by 24%
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
        """Export transaction data to Parquet format with optimized dtypes."""
        df = pd.DataFrame(self.transactions)

        # Ensure correct dtypes for storage efficiency and accuracy
        # All times/latencies are int64 (ms) - no float64 inaccuracy
        # Small counts are int8 to be semantically correct (though compression makes this negligible)
        dtype_map = {
            'txn_id': 'int64',
            't_submit': 'int64',
            't_runtime': 'int64',
            't_commit': 'int64',
            'commit_latency': 'int64',
            'total_latency': 'int64',
            'n_retries': 'int8',
            'n_tables_read': 'int8',
            'n_tables_written': 'int8',
            'status': 'object'  # String type
        }

        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)

        # Also print summary statistics
        print(f"\nSimulation Summary:")
        print(f"  Total transactions: {self.txn_total}")

        if self.txn_total > 0:
            print(f"  Committed: {self.txn_committed} ({100*self.txn_committed/self.txn_total:.1f}%)")
            print(f"  Aborted: {self.txn_aborted} ({100*self.txn_aborted/self.txn_total:.1f}%)")
        else:
            print(f"  Committed: 0")
            print(f"  Aborted: 0")
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

        # Print conflict resolution statistics (CAS mode)
        total_conflicts = self.false_conflicts + self.real_conflicts
        if total_conflicts > 0:
            print(f"\n  Conflict Resolution (CAS mode):")
            print(f"    Total conflicts: {total_conflicts}")
            print(f"    False conflicts: {self.false_conflicts} ({100*self.false_conflicts/total_conflicts:.1f}%)")
            print(f"    Real conflicts: {self.real_conflicts} ({100*self.real_conflicts/total_conflicts:.1f}%)")
            if self.real_conflicts > 0:
                print(f"    Manifest files read: {self.manifest_files_read} (avg {self.manifest_files_read/self.real_conflicts:.1f} per real conflict)")
                print(f"    Manifest files written: {self.manifest_files_written} (avg {self.manifest_files_written/self.real_conflicts:.1f} per real conflict)")

        # Print append mode statistics
        total_appends = self.append_physical_success + self.append_physical_failure
        if total_appends > 0:
            print(f"\n  Append Mode Statistics:")
            print(f"    Total append attempts: {total_appends}")
            print(f"    Physical success: {self.append_physical_success} ({100*self.append_physical_success/total_appends:.1f}%)")
            print(f"    Physical failure: {self.append_physical_failure} ({100*self.append_physical_failure/total_appends:.1f}%)")
            if self.append_physical_success > 0:
                print(f"    Logical success: {self.append_logical_success} ({100*self.append_logical_success/self.append_physical_success:.1f}% of physical success)")
            print(f"    Logical conflicts: {self.append_logical_conflict}")
            print(f"    Compactions triggered: {self.append_compactions_triggered}")
            print(f"    Compactions completed: {self.append_compactions_completed}")
            if self.append_dedup_hits > 0:
                print(f"    Deduplication hits: {self.append_dedup_hits}")