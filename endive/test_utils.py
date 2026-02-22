"""Test utilities for endive simulator.

Provides helper functions for creating test fixtures.
"""

import tempfile
from typing import Optional


def create_test_config(
    output_path: str,
    num_tables: int = 5,
    seed: Optional[int] = 42,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 500.0,
    extra_config: str = "",
    **kwargs,
) -> str:
    """Create a test configuration file with common parameters.

    Args:
        output_path: Path for simulation output
        num_tables: Number of tables in catalog
        seed: Random seed (None for random)
        duration_ms: Simulation duration
        inter_arrival_scale: Mean inter-arrival time
        extra_config: Additional TOML content to append
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

[transaction]
retry = {kwargs.get('retry', 10)}
runtime.mean = {kwargs.get('runtime_mean', 200)}
runtime.sigma = {kwargs.get('runtime_sigma', 1.5)}

inter_arrival.distribution = "exponential"
inter_arrival.scale = {inter_arrival_scale}

operation_types.fast_append = 1.0

[storage]
provider = "{kwargs.get('provider', 'instant')}"
{extra_config}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        return f.name
