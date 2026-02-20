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
