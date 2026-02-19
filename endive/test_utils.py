"""Test utilities for endive simulator.

Provides builder pattern for creating test fixtures with minimal boilerplate.
"""

import numpy as np
import simpy
from typing import Optional
import tempfile
import os

from endive.main import Catalog, Txn, configure_from_toml
import endive.main


class SimulationBuilder:
    """Builder pattern for creating simulation test fixtures.

    Provides a fluent API for constructing simulation environments, catalogs,
    and transactions with sensible defaults, reducing test boilerplate.

    Example:
        env, catalog, txn = (SimulationBuilder()
            .with_tables(5)
            .with_groups(2, "uniform")
            .with_catalog_at_version(10)
            .with_transaction_at_version(5)
            .build())
    """

    def __init__(self):
        self._num_tables = 5
        self._num_groups = 1
        self._group_distribution = "uniform"
        self._catalog_version = 0
        self._txn_version = None  # If None, use catalog version
        self._txn_tables_read = None  # If None, auto-generate
        self._txn_tables_write = None  # If None, auto-generate from read
        self._seed = 42
        self._config_path = None

    def with_tables(self, n: int) -> 'SimulationBuilder':
        """Set number of tables in catalog."""
        self._num_tables = n
        return self

    def with_groups(self, n: int, distribution: str = "uniform") -> 'SimulationBuilder':
        """Set number of table groups and distribution type."""
        self._num_groups = n
        self._group_distribution = distribution
        return self

    def with_catalog_at_version(self, seq: int) -> 'SimulationBuilder':
        """Set initial catalog sequence number."""
        self._catalog_version = seq
        return self

    def with_transaction_at_version(self, seq: int) -> 'SimulationBuilder':
        """Set transaction's captured catalog sequence number."""
        self._txn_version = seq
        return self

    def with_tables_accessed(self, read: dict, write: dict) -> 'SimulationBuilder':
        """Explicitly set tables read and written by transaction.

        Args:
            read: Dict mapping table ID to version read
            write: Dict mapping table ID to version to write
        """
        self._txn_tables_read = read
        self._txn_tables_write = write
        return self

    def with_seed(self, seed: int) -> 'SimulationBuilder':
        """Set random seed for deterministic behavior."""
        self._seed = seed
        return self

    def with_config(self, config_path: str) -> 'SimulationBuilder':
        """Use specific configuration file."""
        self._config_path = config_path
        return self

    def build(self) -> tuple[simpy.Environment, Catalog, Optional[Txn]]:
        """Build and return simulation environment, catalog, and optional transaction.

        Returns:
            Tuple of (SimPy Environment, Catalog, Transaction or None)
        """
        # Load configuration or create minimal one
        if self._config_path:
            configure_from_toml(self._config_path)
        else:
            self._configure_minimal()

        # Set seed for determinism
        np.random.seed(self._seed)

        # Create table grouping
        endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = (
            endive.main.partition_tables_into_groups(
                self._num_tables,
                self._num_groups,
                self._group_distribution,
                {}
            )
        )

        # Create environment and catalog
        env = simpy.Environment()
        catalog = Catalog(env)

        # Set catalog to desired version
        catalog.seq = self._catalog_version
        for i in range(self._num_tables):
            catalog.tbl[i] = self._catalog_version

        # Create transaction if tables are specified or version differs
        txn = None
        if self._txn_tables_read is not None or self._txn_version is not None:
            # Determine transaction version
            txn_ver = self._txn_version if self._txn_version is not None else self._catalog_version

            # Determine tables accessed
            if self._txn_tables_read is not None:
                tblr = self._txn_tables_read
                tblw = self._txn_tables_write if self._txn_tables_write is not None else {}
            else:
                # Auto-generate simple case: read/write table 0
                tblr = {0: txn_ver}
                tblw = {0: txn_ver + 1}

            # Create transaction
            txn = Txn(1, env.now, 100, txn_ver, tblr, tblw)
            txn.v_dirty = {**tblr, **tblw}

        return env, catalog, txn

    def _configure_minimal(self):
        """Configure minimal simulation parameters for testing."""
        endive.main.N_TABLES = self._num_tables
        endive.main.N_GROUPS = self._num_groups
        endive.main.GROUP_SIZE_DIST = self._group_distribution
        endive.main.LONGTAIL_PARAMS = {}
        endive.main.N_TXN_RETRY = 10
        endive.main.MAX_PARALLEL = 4
        endive.main.MIN_LATENCY = 5

        # Simple latency configurations
        endive.main.T_CAS = {'mean': 100, 'stddev': 10}
        endive.main.T_METADATA_ROOT = {
            'read': {'mean': 50, 'stddev': 5},
            'write': {'mean': 60, 'stddev': 6}
        }
        endive.main.T_MANIFEST_LIST = {
            'read': {'mean': 50, 'stddev': 5},
            'write': {'mean': 60, 'stddev': 6}
        }
        endive.main.T_MANIFEST_FILE = {
            'read': {'mean': 50, 'stddev': 5},
            'write': {'mean': 60, 'stddev': 6}
        }

        # PMF placeholders (not used in most unit tests)
        endive.main.N_TBL_PMF = [1.0] + [0.0] * self._num_tables
        endive.main.TBL_R_PMF = [1.0 / self._num_tables] * self._num_tables
        endive.main.N_TBL_W_PMF = [[1.0]] * (self._num_tables + 1)


def create_test_config(
    output_path: str,
    num_tables: int = 5,
    num_groups: int = 1,
    seed: Optional[int] = 42,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 500.0,
    extra_config: str = "",
    **kwargs
) -> str:
    """Create a test configuration file with common parameters.

    Args:
        output_path: Path for simulation output
        num_tables: Number of tables in catalog
        num_groups: Number of table groups
        seed: Random seed (None for random)
        duration_ms: Simulation duration
        inter_arrival_scale: Mean inter-arrival time
        extra_config: Additional TOML content to append (e.g., partition settings)
        **kwargs: Additional config overrides

    Returns:
        Path to created configuration file
    """
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
{f'seed = {seed}' if seed is not None else '# seed = 42'}

[catalog]
num_tables = {num_tables}
num_groups = {num_groups}
group_size_distribution = "{kwargs.get('group_distribution', 'uniform')}"

[transaction]
retry = {kwargs.get('retry', 10)}
runtime.min = {kwargs.get('runtime_min', 100)}
runtime.mean = {kwargs.get('runtime_mean', 200)}
runtime.sigma = {kwargs.get('runtime_sigma', 1.5)}

inter_arrival.distribution = "{kwargs.get('inter_arrival_dist', 'exponential')}"
inter_arrival.scale = {inter_arrival_scale}
inter_arrival.min = 100.0
inter_arrival.max = 1000.0
inter_arrival.mean = 500.0
inter_arrival.std_dev = 100.0
inter_arrival.value = 500.0

ntable.zipf = 2.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

[storage]
max_parallel = {kwargs.get('max_parallel', 4)}
min_latency = {kwargs.get('min_latency', 5)}

T_CAS.mean = {kwargs.get('cas_mean', 100)}
T_CAS.stddev = {kwargs.get('cas_stddev', 10)}

T_METADATA_ROOT.read.mean = 50
T_METADATA_ROOT.read.stddev = 5
T_METADATA_ROOT.write.mean = 60
T_METADATA_ROOT.write.stddev = 6

T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6

T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
{extra_config}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        return f.name
