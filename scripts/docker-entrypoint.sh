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

# Execute the command
exec "$@"
