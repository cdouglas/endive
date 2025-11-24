# Docker Setup for Iceberg Catalog Simulator

This document explains how to run the simulator experiments using Docker.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build the image and run baseline experiments
docker-compose up

# Results will be available in:
# - ./experiments/  - Raw experiment results
# - ./experiment_logs/ - Execution logs
```

### Using Docker Directly

```bash
# Build the image
docker build -t icecap-sim .

# Run baseline experiments
docker run -v $(pwd)/experiments:/app/experiments \
           -v $(pwd)/plots:/app/plots \
           -v $(pwd)/experiment_logs:/app/experiment_logs \
           icecap-sim

# Results will be in ./experiments/
```

## Available Services

### 1. Default: Run Baseline Experiments

Runs all baseline experiments (Exp 2.1 and Exp 2.2).

```bash
docker-compose up
```

**Output:**
- `experiments/exp2_1_*` - Single-table saturation experiments
- `experiments/exp2_2_*` - Multi-table scaling experiments
- `experiment_logs/` - Execution logs with timestamps

**Duration:** ~10-20 hours depending on hardware (63 experiments × 5 seeds each)

### 2. Run Analysis Only

After experiments complete, generate all plots and tables:

```bash
docker-compose run --rm analyze
```

**Output:**
- `plots/exp2_1_analysis/` - Single-table analysis (plots + markdown)
- `plots/exp2_2_analysis/` - Multi-table analysis (plots + markdown)
- `plots/distributions/` - Theoretical distributions (plots + markdown)

**Duration:** ~1-2 minutes

### 3. Run Conformance Tests

Verify that experiment results match configured distributions:

```bash
docker-compose run --rm test
```

**Output:** Test results showing pass/fail for distribution conformance

**Duration:** ~5-10 seconds

### 4. Run Single Experiment

Run a specific experiment configuration:

```bash
# Using environment variable
EXPERIMENT_CONFIG=experiment_configs/exp2_1_single_table_false_conflicts.toml \
  docker-compose run --rm single-experiment

# Or directly with docker
docker run -v $(pwd)/experiments:/app/experiments \
           -v $(pwd)/experiment_configs:/app/experiment_configs:ro \
           icecap-sim \
           bash -c "echo 'Y' | python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml"
```

**Duration:** ~5-60 minutes depending on configuration

## Volume Mounts

The Docker setup uses three volume mounts:

| Host Path | Container Path | Purpose | Access |
|-----------|----------------|---------|--------|
| `./experiments` | `/app/experiments` | Experiment results (.parquet, cfg.toml) | Read/Write |
| `./plots` | `/app/plots` | Analysis plots and tables | Read/Write |
| `./experiment_logs` | `/app/experiment_logs` | Execution logs | Read/Write |
| `./experiment_configs` | `/app/experiment_configs` | Experiment configurations | Read-Only |

## Advanced Usage

### Run Custom Python Command

```bash
docker-compose run --rm icecap-sim python -m icecap.main --help
```

### Run Interactive Shell

```bash
docker run -it --rm \
  -v $(pwd)/experiments:/app/experiments \
  -v $(pwd)/plots:/app/plots \
  icecap-sim bash
```

Inside the container:
```bash
# Run a single experiment
echo "Y" | python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml

# Run analysis
python -m icecap.saturation_analysis -i experiments -o plots/my_analysis -p "exp2_1_*"

# Run tests
pytest tests/ -v
```

### Parallel Experiment Execution

Run multiple containers in parallel (careful with resource usage):

```bash
# Terminal 1: Run exp2_1 experiments
docker run -v $(pwd)/experiments:/app/experiments icecap-sim \
  bash scripts/run_baseline_experiments.sh

# Terminal 2: Meanwhile, analyze completed experiments
docker-compose run --rm analyze
```

### Custom Experiment Configuration

1. Create your config file in `experiment_configs/my_experiment.toml`
2. Run with:
```bash
docker run -v $(pwd)/experiments:/app/experiments \
           -v $(pwd)/experiment_configs:/app/experiment_configs:ro \
           icecap-sim \
           bash -c "echo 'Y' | python -m icecap.main experiment_configs/my_experiment.toml"
```

## Resource Requirements

### Minimum Requirements
- **CPU:** 2 cores
- **RAM:** 4 GB
- **Disk:** 10 GB free space

### Recommended for Full Baseline
- **CPU:** 8+ cores (experiments can run in parallel)
- **RAM:** 16 GB
- **Disk:** 50 GB (experiment results + plots)

## Monitoring Progress

### View Logs in Real-Time

```bash
# If running in background
docker logs -f icecap-experiments

# Or check log files
tail -f experiment_logs/run_*.log
```

### Check Progress

```bash
# Count completed experiments
find experiments -name "results.parquet" | wc -l

# Expected: 315 files (63 experiments × 5 seeds)
```

### Monitor Resource Usage

```bash
docker stats icecap-experiments
```

## Troubleshooting

### Container Exits Immediately

Check logs:
```bash
docker logs icecap-experiments
```

### Out of Memory

Reduce parallel experiments or increase Docker memory limit:
```bash
# Docker Desktop: Settings → Resources → Memory
# Recommended: 8GB minimum
```

### Permission Errors on Volumes

Ensure directories are writable:
```bash
mkdir -p experiments plots experiment_logs
chmod 755 experiments plots experiment_logs
```

### Experiments Taking Too Long

For testing, reduce the number of experiments in `scripts/run_baseline_experiments.sh`:
```bash
# Edit to run only a subset
INTER_ARRIVALS=(1000 500)  # Instead of all 9 values
NUM_TABLES=(1 2)           # Instead of all 6 values
```

## Cleaning Up

### Remove Experiment Results

```bash
rm -rf experiments/* plots/* experiment_logs/*
```

### Remove Docker Images

```bash
docker-compose down --rmi all
# Or
docker rmi icecap-sim
```

### Remove All (Including Volumes)

```bash
docker-compose down -v --rmi all
rm -rf experiments plots experiment_logs
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Run Experiments

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  experiments:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker-compose build

      - name: Run baseline experiments
        run: docker-compose up
        timeout-minutes: 1200  # 20 hours

      - name: Run analysis
        run: docker-compose run --rm analyze

      - name: Run tests
        run: docker-compose run --rm test

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: experiment-results
          path: |
            experiments/
            plots/
          retention-days: 90
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent {
        docker {
            image 'icecap-sim:latest'
            args '-v $WORKSPACE/experiments:/app/experiments -v $WORKSPACE/plots:/app/plots'
        }
    }
    stages {
        stage('Build') {
            steps {
                sh 'docker-compose build'
            }
        }
        stage('Run Experiments') {
            steps {
                sh 'docker-compose up'
            }
        }
        stage('Analyze') {
            steps {
                sh 'docker-compose run --rm analyze'
            }
        }
        stage('Test') {
            steps {
                sh 'docker-compose run --rm test'
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'experiments/**,plots/**', fingerprint: true
        }
    }
}
```

## Performance Tips

1. **Use SSD storage** for experiment results (high I/O)
2. **Allocate more CPU cores** to Docker for parallel execution
3. **Monitor disk space** - experiments generate ~10GB of data
4. **Run analysis separately** after experiments to save time
5. **Use `--no-cache`** when rebuilding after code changes:
   ```bash
   docker-compose build --no-cache
   ```

## Multi-Architecture Support

Build for different platforms:

```bash
# Build for Linux AMD64 (most common)
docker build --platform linux/amd64 -t icecap-sim:amd64 .

# Build for Linux ARM64 (Mac M1/M2)
docker build --platform linux/arm64 -t icecap-sim:arm64 .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t icecap-sim:latest .
```

## Support

For issues with Docker setup:
1. Check logs: `docker logs icecap-experiments`
2. Verify volumes: `docker inspect icecap-experiments`
3. Test interactively: `docker run -it --rm icecap-sim bash`
4. Report issues at: https://github.com/anthropics/claude-code/issues
