#!/bin/bash
#
# Prepare configuration bundle for missing experiments
#
# This script scans the local experiments directory, identifies experiments
# with missing results, and exports their configurations to a portable bundle
# that can be copied to a remote machine for execution.
#
# Usage:
#   ./scripts/prepare_missing_experiments.sh [--output DIR] [--expected-seeds N]
#
# Options:
#   --output DIR         Output directory for bundle (default: experiment_batch)
#   --expected-seeds N   Expected number of seeds per experiment (default: 5)
#

set -e

# Parse arguments
OUTPUT_DIR="experiment_batch"
EXPECTED_SEEDS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --expected-seeds)
            EXPECTED_SEEDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output DIR] [--expected-seeds N]"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Preparing Missing Experiments Bundle"
echo "========================================="
echo "Expected seeds per experiment: $EXPECTED_SEEDS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create clean output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Warning: Output directory exists, removing..."
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR/configs"

# Create manifest file
MANIFEST="$OUTPUT_DIR/manifest.txt"
touch "$MANIFEST"

echo "Scanning experiments directory..."

# Find all experiment directories with missing results
experiment_count=0
total_simulations=0

for exp_dir in experiments/exp*-[0-9a-f]*; do
    # Skip if not a directory
    if [ ! -d "$exp_dir" ]; then
        continue
    fi

    # Count results.parquet files
    result_count=$(find "$exp_dir" -name results.parquet 2>/dev/null | wc -l)

    if [ "$result_count" -lt "$EXPECTED_SEEDS" ]; then
        # Check if cfg.toml exists
        if [ ! -f "$exp_dir/cfg.toml" ]; then
            echo "  Warning: No cfg.toml in $exp_dir, skipping"
            continue
        fi

        exp_name=$(basename "$exp_dir")
        missing_count=$((EXPECTED_SEEDS - result_count))
        echo "  $exp_name: $result_count/$EXPECTED_SEEDS results (missing $missing_count)"

        # Create directory in bundle
        bundle_exp_dir="$OUTPUT_DIR/configs/$exp_name"
        mkdir -p "$bundle_exp_dir"

        # Copy config
        cp "$exp_dir/cfg.toml" "$bundle_exp_dir/cfg.toml"

        # Find and record missing seeds
        for seed_dir in "$exp_dir"/[0-9]*; do
            if [ -d "$seed_dir" ]; then
                seed=$(basename "$seed_dir")
                if [ ! -f "$seed_dir/results.parquet" ]; then
                    echo "$exp_name:$seed" >> "$MANIFEST"
                    total_simulations=$((total_simulations + 1))
                fi
            fi
        done

        experiment_count=$((experiment_count + 1))
    fi
done

if [ $experiment_count -eq 0 ]; then
    echo ""
    echo "No missing experiments found."
    rm -rf "$OUTPUT_DIR"
    exit 0
fi

echo ""
echo "Summary:"
echo "  Experiments with missing results: $experiment_count"
echo "  Total simulations to run: $total_simulations"
echo ""

# Create execution script
EXEC_SCRIPT="$OUTPUT_DIR/run_experiments.sh"
cat > "$EXEC_SCRIPT" << 'EXEC_EOF'
#!/bin/bash
#
# Execute experiment batch on remote machine
#
# This script is generated automatically by prepare_missing_experiments.sh
# It reads the manifest and runs all missing simulations in parallel.
#
# Usage:
#   ./run_experiments.sh [--parallel N] [--output DIR]
#
# Options:
#   --parallel N    Number of parallel simulations (default: 4)
#   --output DIR    Output directory for results (default: results)
#

set -e

# Parse arguments
PARALLEL=4
OUTPUT_DIR="results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Running Experiment Batch"
echo "========================================="
echo "Parallel simulations: $PARALLEL"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MANIFEST="$SCRIPT_DIR/manifest.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "Error: manifest.txt not found"
    exit 1
fi

# Count total simulations
TOTAL=$(wc -l < "$MANIFEST")
echo "Total simulations to run: $TOTAL"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a single simulation
run_simulation() {
    local exp_name="$1"
    local seed="$2"
    local script_dir="$3"
    local output_base="$4"

    local cfg_path="$script_dir/configs/$exp_name/cfg.toml"
    local exp_output_dir="$output_base/$exp_name"
    local seed_output_dir="$exp_output_dir/$seed"

    mkdir -p "$seed_output_dir"

    echo "[$(date '+%H:%M:%S')] Starting: $exp_name seed=$seed"

    # Run simulation
    cd "$seed_output_dir"
    python -m icecap.main "$cfg_path" --seed "$seed" > run.log 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ] && [ -f "results.parquet" ]; then
        echo "[$(date '+%H:%M:%S')] ✓ Completed: $exp_name seed=$seed"
        # Remove log file on success to save space
        rm -f run.log
        return 0
    else
        echo "[$(date '+%H:%M:%S')] ✗ Failed: $exp_name seed=$seed (exit code: $exit_code)"
        # Keep log file for debugging
        return 1
    fi
}

export -f run_simulation

# Run simulations in parallel
completed=0
failed=0

while IFS=':' read -r exp_name seed; do
    # Wait for available slot
    while [ $(jobs -r | wc -l) -ge "$PARALLEL" ]; do
        sleep 1
    done

    # Start simulation in background
    run_simulation "$exp_name" "$seed" "$SCRIPT_DIR" "$OUTPUT_DIR" &
done < "$MANIFEST"

# Wait for all jobs to complete
wait

# Count results
for line in $(cat "$MANIFEST"); do
    exp_name="${line%:*}"
    seed="${line#*:}"
    if [ -f "$OUTPUT_DIR/$exp_name/$seed/results.parquet" ]; then
        completed=$((completed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo "========================================="
echo "Batch Execution Complete"
echo "========================================="
echo "Total simulations: $TOTAL"
echo "Completed successfully: $completed"
echo "Failed: $failed"
echo ""

if [ $failed -gt 0 ]; then
    echo "Some simulations failed. Check run.log files in output directories."
    exit 1
fi

echo "Results written to: $OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. Copy $OUTPUT_DIR/ back to local machine"
echo "  2. Run merge script to integrate into experiments/ directory"
EXEC_EOF

chmod +x "$EXEC_SCRIPT"

# Create README
README="$OUTPUT_DIR/README.md"
cat > "$README" << 'README_EOF'
# Experiment Batch Bundle

This bundle contains configurations for experiments with missing results.

## Contents

- `manifest.txt` - List of experiment:seed pairs to run
- `configs/` - Configuration files (cfg.toml) for each experiment
- `run_experiments.sh` - Execution script for remote machine

## Workflow

### 1. Copy to remote machine

```bash
tar czf experiment_batch.tar.gz experiment_batch/
scp experiment_batch.tar.gz remote-machine:~/
```

### 2. On remote machine

```bash
tar xzf experiment_batch.tar.gz
cd experiment_batch
./run_experiments.sh --parallel 8
```

### 3. Copy results back

```bash
tar czf results.tar.gz results/
scp remote-machine:~/experiment_batch/results.tar.gz .
```

### 4. Merge results locally

```bash
tar xzf results.tar.gz
./scripts/merge_experiment_results.sh results/
```

## Configuration

- Adjust parallelism: `./run_experiments.sh --parallel N`
- Each simulation takes ~1 hour
- Monitor progress: `tail -f results/*/*/run.log`

README_EOF

echo "Bundle created successfully!"
echo ""
echo "Contents:"
echo "  - manifest.txt ($total_simulations simulations)"
echo "  - configs/ ($experiment_count experiment configs)"
echo "  - run_experiments.sh (execution script)"
echo "  - README.md (instructions)"
echo ""
echo "Next steps:"
echo "  1. Review the bundle: ls -lah $OUTPUT_DIR/"
echo "  2. Create tarball: tar czf $OUTPUT_DIR.tar.gz $OUTPUT_DIR/"
echo "  3. Copy to remote machine"
echo "  4. Run: cd $OUTPUT_DIR && ./run_experiments.sh --parallel 8"
