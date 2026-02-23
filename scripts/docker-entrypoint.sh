#!/bin/bash
# Entrypoint for dev container - ensures venv is ready

set -e

VENV_DIR="/app/.venv"

# Check if venv needs to be created or repaired
# Test pip specifically since scripts have hardcoded shebangs from host
needs_repair=false
if [ ! -x "$VENV_DIR/bin/python" ]; then
    needs_repair=true
elif ! "$VENV_DIR/bin/pip" --version &>/dev/null; then
    needs_repair=true
fi

if $needs_repair; then
    echo "Creating/repairing venv..."
    rm -rf "$VENV_DIR"
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet -r /app/requirements.txt
    "$VENV_DIR/bin/pip" install --quiet -e /app
    echo "venv ready."
fi

# Bridge host/container home directories.
# Claude Code plugin JSON files (installed_plugins.json, known_marketplaces.json)
# store absolute paths using the host $HOME (e.g. /home/chris/.claude/...).
# Inside the container the home is /home/dev, so those paths don't resolve.
# A symlink from the host home to the container home makes them work.
if [ -n "$HOST_HOME" ] && [ "$HOST_HOME" != "$HOME" ] && [ ! -e "$HOST_HOME" ]; then
    mkdir -p "$(dirname "$HOST_HOME")" 2>/dev/null &&
    ln -sf "$HOME" "$HOST_HOME" 2>/dev/null || true
fi

# Execute the command
exec "$@"
