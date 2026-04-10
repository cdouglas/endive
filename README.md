# Endive: Iceberg Catalog Simulator

Discrete-event simulator for Apache Iceberg's optimistic concurrency control (OCC). Models catalog contention, conflict resolution, and commit latency across cloud storage providers to answer: **When does commit throughput saturate? What causes latency to explode?**

## What it models

Iceberg installs new table versions via a compare-and-swap (CAS) on the catalog. When CAS fails but the transaction commutes with the new state, it merges metadata and retries. Endive models:

- **CAS and append catalogs** backed by opaque `StorageProvider`s (`CASCatalog`, `AppendCatalog`, `InstantCatalog`)
- **Transaction types** with distinct conflict behavior: `FastAppendTransaction` (always retries) and `ValidatedOverwriteTransaction` (aborts on real conflict, pays historical manifest-list reads as an I/O convoy)
- **Per-partition version tracking** so concurrent writers to disjoint partitions retry for free (catalog read + CAS only, no manifest I/O)
- **Table metadata inlining** — store table metadata inside the catalog CAS object to eliminate per-attempt metadata I/O at the cost of CAS payload size
- **ML+ (manifest-list append) mode** — avoid ML rewrites on false conflicts where the storage layer supports conditional append
- **Provider-calibrated latencies** from YCSB June 2025 benchmarks (S3, S3 Express, Azure, Azure Premium, GCS) and Durner et al. VLDB 2023 PUT/GET measurements

Complete API and invariants: [SPEC.md](SPEC.md). Model simplifications: [docs/model.md](docs/model.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Run a single 1-hour simulation (~4 seconds wall-clock)
python -m endive.main experiment_configs/exp1_fa_baseline.toml --yes

# Run the full test suite
pytest tests/ -q
```

## Running experiments

```bash
# Run all experiments in parallel with 3 seeds per config
python scripts/run_all_experiments.py --parallel 4 --seeds 3

# Run specific groups
python scripts/run_all_experiments.py --groups baseline,heatmap,catalog --seeds 3

# Quick smoke test (1-minute sims, 3 load levels)
python scripts/run_all_experiments.py --quick --parallel 4
```

Experiment groups defined in `scripts/run_all_experiments.py`:

| Group | Configs | Purpose |
|-------|---------|---------|
| `baseline` | `exp1_fa_baseline` | 100% FastAppend saturation curve |
| `heatmap` | `exp2_mix_heatmap` | FA/VO mix × arrival-rate heatmap |
| `catalog` | `exp3a_catalog_fa`, `exp3b_catalog_mix` | Catalog CAS latency impact |
| `tables` | `exp4a_tables_fa`, `exp4b_tables_mix` | Multi-table contention (uniform selection) |
| `zipf` | `exp4a_zipf_tables_fa`, `exp4b_zipf_tables_mix` | Zipf table selection |
| `providers` | `exp4c_tables_providers` | Real provider profiles × tables × workload |
| `partition` | `exp5[ab]_[zipf_]partition_{fa,mix}` | Per-partition conflict detection ([docs/EXP5.md](docs/EXP5.md)) |
| `inlined` | `exp6[ab]_[zipf_]inlined_{fa,mix}` | Inlined table metadata ([docs/EXP6.md](docs/EXP6.md)) |

Results land in `experiments/<label>-<hash>/<seed>/results.parquet` plus a consolidated `experiments/consolidated.parquet` (see [docs/CONSOLIDATED_FORMAT.md](docs/CONSOLIDATED_FORMAT.md)).

## Analysis

```bash
# Regenerate every plot declared in experiment configs
python scripts/regenerate_plots.py

# Preview what would be generated
python scripts/regenerate_plots.py --dry-run

# Ad-hoc analysis against the consolidated results
python -m endive.saturation_analysis -i experiments -p "exp1_fa_baseline-*" -o plots/exp1_fa_baseline
python -m endive.saturation_analysis -i experiments -p "exp4a_tables_fa-*" -o plots/exp4a --group-by num_tables
```

Filter syntax (AND logic via repeated `--filter`):

```bash
python -m endive.saturation_analysis -i experiments -p "exp4b_*" \
    -o plots/exp4b_t10 --group-by real_conflict_probability \
    --filter "num_tables==10"
```

## Storage provider latencies

Calibrated from YCSB June 2025 benchmarks and Durner et al. VLDB 2023. See [docs/analysis/latency_verification.md](docs/analysis/latency_verification.md) for sources.

| Provider | CAS median | Append median | PUT base | PUT rate |
|----------|------------|---------------|----------|----------|
| S3 Standard   | 61 ms | —      | 60 ms | 20 ms/MiB |
| S3 Express    | 22 ms | 21 ms  | 6.5 ms | 10 ms/MiB |
| Azure Std     | 93 ms | 87 ms  | 45 ms | 25 ms/MiB |
| Azure Premium | 64 ms | 70 ms  | 41 ms | 15 ms/MiB |
| GCP           | 170 ms | —     | 200 ms | 17 ms/MiB |

S3 Standard and GCS lack conditional append and are excluded from append experiments.

## Module layout

```
endive/
├── storage.py            # StorageProvider ABC, latency distributions, provider implementations
├── catalog.py            # Catalog ABC, CASCatalog, AppendCatalog, InstantCatalog
├── transaction.py        # Transaction ABC, FastAppend, ValidatedOverwrite, commit loop
├── conflict_detector.py  # Probabilistic and PartitionOverlap detectors
├── workload.py           # Workload generator, table/partition selectors
├── simulation.py         # SimPy runner, SimulationConfig, Statistics (streaming parquet)
├── config.py             # TOML loading, provider profiles, validation
├── main.py               # CLI entry point, experiment directory management
└── saturation_analysis.py  # Analysis + plotting pipeline
```

Design invariant: all latency-bearing operations yield bare `float` milliseconds; the only place SimPy is used is `Simulation._drive_generator()`. Transactions observe catalog state exclusively through immutable `CatalogSnapshot`s returned by `catalog.read()`.

## Documentation

- **[SPEC.md](SPEC.md)** — authoritative module APIs, invariants, I/O cost model, TOML schema
- **[docs/model.md](docs/model.md)** — simplifications relative to real Iceberg
- **[docs/EXP5.md](docs/EXP5.md)**, **[docs/EXP6.md](docs/EXP6.md)** — partition-aware and inlined-metadata experiment designs
- **[docs/CONSOLIDATED_FORMAT.md](docs/CONSOLIDATED_FORMAT.md)** — consolidated parquet output format
- **[docs/analysis/](docs/analysis/)** — provider latency verification, DES profiling, reference data
- **[docs/README.md](docs/README.md)** — documentation index

## References

- Apache Iceberg: <https://iceberg.apache.org/>
- Durner, Leis, Neumann. *Exploiting Cloud Object Storage for High-Performance Analytics*. VLDB 2023.
- SimPy: <https://simpy.readthedocs.io/>
