# Quick Start Guide

## Setup

```bash
# Activate virtual environment
source bin/activate

# Verify installation
python -m icecap.main --help
```

## Run Your First Experiment

### 1. Explore Client Load Impact

Generate CDF and success rate plots for different client loads (inter-arrival times):

```bash
# Run experiments with 6 different client loads
python -m icecap.experiment -o experiments/clients sweep-clients \
    --times 100 500 1000 2000 5000 10000 \
    --dist exponential

# Generate plots
python -m icecap.analysis -i experiments/clients -o plots/clients all
```

**What to look for:**
- Lower inter-arrival time (100ms) = high client load = more conflicts
- Higher inter-arrival time (10000ms) = low client load = fewer conflicts
- Check `plots/clients/cdf_commit_latency_clients.png` to see latency distribution
- Check `plots/clients/success_rate_clients.png` to see abort rates

### 2. Explore Catalog Latency Impact

Generate plots showing how catalog latency affects commit times:

```bash
# Run experiments with 6 different CAS latencies
python -m icecap.experiment -o experiments/latency sweep-latency \
    --latencies 10 50 100 200 500 1000

# Generate plots
python -m icecap.analysis -i experiments/latency -o plots/latency all
```

**What to look for:**
- Higher CAS latency amplifies retry costs
- Check `plots/latency/cdf_commit_latency_cas.png`

### 3. Combined Analysis

Run experiments varying both parameters for comprehensive analysis:

```bash
# Run combined sweep (9 experiments: 3 client loads × 3 latencies)
python -m icecap.experiment -o experiments/combined sweep-combined \
    --times 500 1000 5000 \
    --latencies 50 100 200

# Generate comprehensive plots
python -m icecap.analysis -i experiments/combined -o plots/combined all
```

**What to look for:**
- `plots/combined/catalog_latency_impact.png` shows 4-panel analysis:
  - How mean and P95 commit latency scale with CAS latency
  - How retry rates change
  - How success rates are affected
- Each line represents a different client load

## Customize Configuration

Edit `cfg.toml` to adjust:

```toml
[simulation]
duration_ms = 100000000  # Increase for more stable statistics

[transaction]
retry = 10               # Increase to reduce aborts (longer commit times)

[storage]
T_CAS = 100             # Adjust catalog operation latency
T_MANIFEST_FILE = 100   # Adjust manifest operation latency
```

## Interpret Results

### Success Rate vs Inter-Arrival Time
- **High success rate (>95%)**: System is under-loaded, most transactions succeed first try
- **Medium success rate (70-95%)**: System is moderately loaded, some retries needed
- **Low success rate (<70%)**: System is overloaded, many aborts even after retries

### Commit Latency CDF
- **Steep initial rise**: Most transactions commit quickly
- **Long tail**: Some transactions require many retries
- **Multiple steps in CDF**: Discrete retry attempts (each retry adds ~RTT)

### Optimal Operating Point
Look for the "knee" in the throughput curve:
- Before the knee: increasing load increases throughput
- After the knee: increasing load decreases throughput (more aborts)

## Markov Model Comparison

For quick analytical estimates without running simulations:

```bash
# Run analytical model
python plot_rtt.py
```

This generates `rtt.png` showing expected transaction time vs number of retries.
Use this to:
- Quickly estimate impact of retry limits
- Validate simulator results
- Understand theoretical best-case performance

## Next Steps

1. **Vary table distributions**: Edit `ntable.zipf`, `seltbl.zipf`, `seltblw.zipf` in config
2. **Test different arrival patterns**: Try `"fixed"` or `"uniform"` distributions
3. **Model different storage systems**: Adjust latency parameters in `[storage]`
4. **Scale up**: Increase `num_tables` and `duration_ms` for large-scale simulations
