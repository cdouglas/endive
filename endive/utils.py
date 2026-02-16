"""Utility functions for the Endive simulator."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def get_git_sha() -> str:
    """Get the current git SHA, or return 'unknown' if not in a git repo.

    Checks in order:
    1. GIT_SHA environment variable (set by Docker build)
    2. Git command (if in a git repository)
    3. Returns 'unknown' if neither available
    """
    # Check environment variable first (set by Docker)
    env_sha = os.environ.get('GIT_SHA')
    if env_sha:
        return env_sha

    # Try git command
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return 'unknown'


def get_git_sha_short() -> str:
    """Get the short (7 char) git SHA."""
    sha = get_git_sha()
    return sha[:7] if sha != 'unknown' else 'unknown'


def partition_tables_into_groups(n_tables: int, n_groups: int, distribution: str, longtail_params: dict) -> tuple[dict, dict]:
    """Partition tables into groups.

    Returns:
        table_to_group: dict mapping table ID to group ID
        group_to_tables: dict mapping group ID to list of table IDs
    """
    if n_groups > n_tables:
        logger.warning(f"Number of groups ({n_groups}) exceeds number of tables ({n_tables}). Setting n_groups = n_tables.")
        n_groups = n_tables

    table_to_group = {}
    group_to_tables = {g: [] for g in range(n_groups)}

    if distribution == "uniform":
        # Uniform distribution: distribute tables evenly across groups
        tables_per_group = n_tables // n_groups
        remainder = n_tables % n_groups

        table_id = 0
        for group_id in range(n_groups):
            # Give extra table to first 'remainder' groups
            group_size = tables_per_group + (1 if group_id < remainder else 0)
            for _ in range(group_size):
                table_to_group[table_id] = group_id
                group_to_tables[group_id].append(table_id)
                table_id += 1

    elif distribution == "longtail":
        # Longtail distribution: one large group, few medium groups, many small groups
        large_frac = longtail_params.get("large_group_fraction", 0.5)
        medium_count = longtail_params.get("medium_groups_count", 3)
        medium_frac = longtail_params.get("medium_group_fraction", 0.3)

        # Calculate sizes
        large_size = int(n_tables * large_frac)
        remaining_after_large = n_tables - large_size
        medium_total_size = int(remaining_after_large * medium_frac)
        small_total_size = remaining_after_large - medium_total_size

        # Ensure we have enough groups
        if n_groups < 1 + medium_count:
            logger.warning(f"Not enough groups ({n_groups}) for longtail distribution (need at least {1 + medium_count}). Using uniform distribution.")
            return partition_tables_into_groups(n_tables, n_groups, "uniform", {})

        medium_size = medium_total_size // medium_count if medium_count > 0 else 0
        small_groups_count = n_groups - 1 - medium_count
        small_size = small_total_size // small_groups_count if small_groups_count > 0 else 0

        table_id = 0
        group_id = 0

        # Large group
        for _ in range(large_size):
            table_to_group[table_id] = group_id
            group_to_tables[group_id].append(table_id)
            table_id += 1
        group_id += 1

        # Medium groups
        for _ in range(medium_count):
            for _ in range(medium_size):
                if table_id < n_tables:
                    table_to_group[table_id] = group_id
                    group_to_tables[group_id].append(table_id)
                    table_id += 1
            group_id += 1

        # Small groups (distribute remaining tables)
        while table_id < n_tables:
            for gid in range(group_id, n_groups):
                if table_id < n_tables:
                    table_to_group[table_id] = gid
                    group_to_tables[gid].append(table_id)
                    table_id += 1

    else:
        raise ValueError(f"Unknown group size distribution: {distribution}")

    return table_to_group, group_to_tables
