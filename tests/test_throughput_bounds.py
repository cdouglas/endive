"""Theoretical throughput upper-bound tests.

These tests catch simulator bugs where information flows between client
and server without a matching message delay — any such leak allows the
simulator to "pack" more commits per unit time than physics permits.

Methodology
-----------

With a constant-latency storage provider where every I/O operation
takes exactly L milliseconds, we can compute the minimum time between
successive successful commits in a single-partition workload from
first principles.

For non-inlined table metadata, each successful commit requires:

    Read catalog (observe previous commit) →  L/2  (half-RTT to see state)
    Read table metadata (per-attempt)      →  L
    Read manifest list  (per-attempt)      →  L
    Write manifest list (per-attempt)      →  L
    Write table metadata (per-attempt)     →  L
    CAS (half-RTT to server check)         →  L/2
                                              ────
                                              5L

Therefore the upper bound on committed throughput is 1 / (5L). Any
observed rate above this is a bug — information is flowing without
a message delay somewhere.

For inlined table metadata, TM reads/writes are eliminated:

    Read catalog                           →  L/2
    Read manifest list                     →  L
    Write manifest list                    →  L
    CAS                                    →  L/2
                                              ────
                                              3L

Upper bound: 1 / (3L).

For ML+ mode (append instead of rewrite), the ML write is eliminated:

    Read catalog                           →  L/2
    Read table metadata                    →  L
    Read manifest list                     →  L
    Write table metadata                   →  L
    CAS                                    →  L/2
                                              ────
                                              4L

Upper bound: 1 / (4L).

These bounds assume a single partition per transaction (so all conflicts
are overlapping). With N partitions and disjoint writes, the bound
changes because disjoint retries are free.
"""

import numpy as np
import pytest

from endive.catalog import InstantCatalog
from endive.conflict_detector import ProbabilisticConflictDetector
from endive.simulation import Simulation, SimulationConfig
from endive.storage import InstantStorageProvider, LognormalLatency
from endive.workload import Workload, WorkloadConfig


def _run_saturated(
    L: float,
    duration_ms: float = 5_000.0,
    metadata_inlined: bool = False,
    ml_append_mode: bool = False,
    num_partitions: int = 1,
    seed: int = 42,
) -> tuple[int, float]:
    """Run a saturated single-table single-partition-per-txn workload.

    Returns (committed_count, observed_rate_per_second).
    """
    # Inter-arrival WELL below the theoretical max to fully saturate.
    # Target 50x the theoretical max rate — heavy over-saturation
    # maximizes the chance of exposing any "free information" leaks.
    target_rate = 50.0 / (5.0 * L / 1000.0)  # txns/sec
    ia_ms = 1000.0 / target_rate
    wl_cfg = WorkloadConfig(
        inter_arrival=LognormalLatency(
            mu=float(np.log(ia_ms)), sigma=0.001, min_latency_ms=0.1,
        ),
        runtime=LognormalLatency(
            mu=float(np.log(1.0)), sigma=0.001, min_latency_ms=0.1,
        ),
        num_tables=1,
        partitions_per_table=(num_partitions,),
        fast_append_weight=1.0,
        validated_overwrite_weight=0.0,
        partitions_per_txn=1,
    )
    workload = Workload(wl_cfg, seed=seed)

    config = SimulationConfig(
        duration_ms=duration_ms,
        seed=seed,
        storage_provider=InstantStorageProvider(
            rng=np.random.RandomState(seed), latency_ms=L,
        ),
        catalog=InstantCatalog(
            num_tables=1, partitions_per_table=(num_partitions,),
            latency_ms=L,
        ),
        workload=workload,
        conflict_detector=ProbabilisticConflictDetector(
            0.0, rng=np.random.RandomState(seed),
        ),
        max_retries=20,
        ml_append_mode=ml_append_mode,
        metadata_inlined=metadata_inlined,
    )
    stats = Simulation(config).run()
    duration_s = duration_ms / 1000.0
    return stats.committed, stats.committed / duration_s


# ---------------------------------------------------------------------------
# Non-inlined: max = 1 / (5L)
# ---------------------------------------------------------------------------

class TestNonInlinedThroughputBound:
    """Single-partition, non-inlined commits cannot exceed 1/(5L) c/s."""

    @pytest.mark.parametrize("L", [50.0, 100.0])
    def test_single_partition_fa_bound(self, L):
        """FA with constant L-ms latency: throughput must not exceed 1/(5L)."""
        # Scale duration with L to ensure enough commits for stable rate
        dur_ms = max(10_000.0, 200.0 * L)
        committed, observed = _run_saturated(L=L, duration_ms=dur_ms)

        max_rate = 1.0 / (5 * L / 1000.0)  # commits per second

        # Tight 3% upper bound: any information leak that bypasses a
        # message delay pushes the rate above the theoretical max.
        assert observed <= max_rate * 1.03, (
            f"L={L}ms: observed {observed:.2f} c/s exceeds "
            f"theoretical upper bound {max_rate:.2f} c/s "
            f"(5L = {5 * L}ms). Committed={committed}. "
            f"This indicates information is flowing without a message "
            f"delay somewhere in the commit protocol."
        )
        # Tight 88% lower bound: any bogus extra I/O (e.g., an extra
        # op per attempt) drops the saturated rate below this.
        # The correct model saturates at ~95% of the theoretical max;
        # a single extra op drops it to ~83% (1/6L vs 1/5L).
        assert observed >= max_rate * 0.88, (
            f"L={L}ms: observed {observed:.2f} c/s is below 88% of "
            f"bound {max_rate:.2f} c/s. Either the workload isn't "
            f"saturated, or there's spurious I/O inflating per-attempt "
            f"cost. Committed={committed}."
        )

    def test_single_partition_fa_bound_50ms(self):
        """At L=50ms, max throughput is 4 c/s (1 commit per 250ms)."""
        committed, observed = _run_saturated(L=50.0, duration_ms=10_000.0)
        # Upper: no information leaks (observed < 4.12)
        assert observed <= 4.12, (
            f"L=50ms: observed {observed:.2f} c/s exceeds 4.12 c/s "
            f"(theoretical 4.0 + 3% tolerance)"
        )
        # Lower: no spurious I/O (observed > 3.52 = 88% of 4.0)
        assert observed >= 3.52, (
            f"L=50ms: observed {observed:.2f} c/s below 3.52 c/s "
            f"(theoretical 4.0 × 88%). Possible spurious per-attempt I/O."
        )


# ---------------------------------------------------------------------------
# Inlined: max = 1 / (3L)
# ---------------------------------------------------------------------------

class TestInlinedThroughputBound:
    """Single-partition, inlined commits cannot exceed 1/(3L) c/s.

    Inlined eliminates TM read/write (2 ops per attempt), but the
    catalog CAS payload is larger. With a constant-latency provider,
    size doesn't matter — all ops are L ms.
    """

    @pytest.mark.parametrize("L", [50.0])
    def test_single_partition_fa_inlined_bound(self, L):
        committed, observed = _run_saturated(
            L=L, duration_ms=10_000.0, metadata_inlined=True,
        )
        max_rate = 1.0 / (3 * L / 1000.0)
        assert observed <= max_rate * 1.03, (
            f"L={L}ms inlined: observed {observed:.2f} c/s exceeds "
            f"theoretical upper bound {max_rate:.2f} c/s "
            f"(3L = {3 * L}ms). Committed={committed}."
        )
        # Tight lower bound catches spurious I/O regressions.
        # Correct model saturates at ~92% (3 ops → cycle 3L is tighter);
        # 1 extra op would drop to ~75% (cycle 4L vs 3L).
        assert observed >= max_rate * 0.82, (
            f"L={L}ms inlined: observed {observed:.2f} c/s below 82% "
            f"of bound {max_rate:.2f}. Possible spurious I/O. "
            f"Committed={committed}."
        )


# ---------------------------------------------------------------------------
# ML+ mode: max = 1 / (4L)
# ---------------------------------------------------------------------------

class TestMLAppendThroughputBound:
    """Single-partition, ML+ mode cannot exceed 1/(4L) c/s.

    ML+ mode eliminates ML writes (the append replaces the rewrite),
    so per-attempt has TM_read + ML_read + TM_write = 3 ops.
    Plus L/2 + L/2 for read and CAS half-RTTs: total 4L.
    """

    def test_single_partition_fa_ml_append_bound_50ms(self):
        committed, observed = _run_saturated(
            L=50.0, duration_ms=10_000.0, ml_append_mode=True,
        )
        max_rate = 1.0 / (4 * 50.0 / 1000.0)  # 5 c/s
        assert observed <= max_rate * 1.03, (
            f"ML+ L=50ms: observed {observed:.2f} c/s exceeds bound "
            f"{max_rate:.2f} c/s (4L = 200ms). Committed={committed}."
        )
        assert observed >= max_rate * 0.85, (
            f"ML+ L=50ms: observed {observed:.2f} c/s below 85% of "
            f"bound {max_rate:.2f}. Possible spurious I/O. "
            f"Committed={committed}."
        )


# ---------------------------------------------------------------------------
# Multi-partition: disjoint retries are free, so the bound is different
# ---------------------------------------------------------------------------

class TestMultiPartitionThroughputBound:
    """With N partitions and ppt=1, disjoint retries skip per-attempt I/O.

    The throughput is NOT bounded by 1/(5L) because multiple transactions
    can have non-overlapping partitions. The bound becomes effectively
    the CAS rate: 1/L per server-side CAS.

    We still test that throughput doesn't exceed this looser bound.
    """

    def test_32_partitions_fa_bound_50ms(self):
        """32 partitions, ppt=1 at L=50ms: bound is higher than single-partition."""
        committed, observed = _run_saturated(
            L=50.0, duration_ms=10_000.0, num_partitions=32,
        )
        # With 32 partitions and random selection, overlap probability is
        # ~3% per pair. Most retries are disjoint (free). The bound is
        # roughly the CAS serialization rate: 1/L = 20 c/s.
        # Allow generous tolerance.
        cas_rate = 1.0 / (50.0 / 1000.0)  # 20 c/s
        assert observed <= cas_rate * 1.20, (
            f"32 partitions L=50ms: observed {observed:.2f} c/s exceeds "
            f"CAS rate bound {cas_rate:.2f} c/s. Committed={committed}."
        )
        # Should beat the single-partition bound (since disjoint retries
        # are mostly free)
        single_partition_bound = 1.0 / (5 * 50.0 / 1000.0)  # 4 c/s
        assert observed > single_partition_bound, (
            f"32 partitions L=50ms: observed {observed:.2f} c/s should "
            f"exceed single-partition bound {single_partition_bound:.2f} "
            f"c/s (disjoint retries are free)"
        )
