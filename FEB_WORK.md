# February 2026 Work Report - Endive Simulator

## Summary

The simulator was rewritten from a monolithic `main.py` (~1,700 lines) to a modular architecture: separate modules for storage, catalog, transactions, conflict detection, workload generation, and simulation orchestration. A formal specification (SPEC.md) guided the rewrite. The test suite grew from ~136 to 500+ tests.

Latency modeling was replaced. Constant-time approximations gave way to lognormal distributions calibrated against June 2025 YCSB cloud storage benchmarks. Provider profiles (S3, S3 Express, Azure Blob, Azure Premium, GCP) are now loaded from TOML files. Catalog CAS latency and storage I/O latency are configured independently.

Experiment infrastructure was consolidated. A unified runner handles multi-dimensional parameter sweeps, deterministic seed generation, Docker-based parallel execution, and parquet consolidation. Plot regeneration reads declarative `[plots]` sections from experiment configs, with per-graph filters, output suffixes, and parallel processing.

The exp4 series swept multi-table contention under single-file catalogs. Exp4a/4b used artificial catalog latencies; exp4c replaced these with real storage provider profiles (S3 Express, S3, Azure Premium, Azure, GCP) in a 4D sweep: provider × num_tables × FA/VO ratio × arrival rate (900 configurations, 5 seeds each). Maximum commit throughput is bounded by 1/CAS_latency - ~45 c/s for S3 Express (22ms CAS) down to ~6 c/s for GCP (170ms CAS). Multi-table configs improve throughput via cross-table retry optimization (skip per-attempt I/O). ValidatedOverwrite P99 latency reaches 245 seconds at saturation while FastAppend stays under 1 second.

---

## Table of Contents

- [Week of Feb 22-28](#week-of-feb-2228) - Full architectural rewrite, experiment infrastructure, exp4 analysis and real-provider sweep, latency validation, consolidation improvements
- [Week of Feb 15-21](#week-of-feb-1521) - Rewrite preparation: CAS modeling, groups removal, table-level conflict detection, config extraction, partition scaling
- [Week of Feb 8-14](#week-of-feb-814) - Realistic latency modeling, partition-level simulation, factorial experiment design
- [Week of Feb 1-7](#week-of-feb-17) - Manifest list append protocol and initial experiment configs

---

## Week of Feb 22-28

Completed the full architectural rewrite on Feb 22: all seven SPEC.md modules (StorageProvider, Catalog, Transaction types, ConflictDetector, Workload, Simulation runner, TOML config loading) implemented and old main.py replaced. Built unified experiment infrastructure on Feb 23 with declarative `[plots]` configs, streaming parquet export, and per-operation-type heatmaps. The rest of the week focused on exp4: fixed exp4a/4b heatmaps mixing catalog latencies, added filter support, parallelized plot regeneration, fixed cross-table CAS retry semantics, and added hatched xheatmap variants. Designed and ran exp4c (5 providers × 6 table counts × 3 FA/VO ratios × 10 arrival rates, 4500 runs). Validated provider latency profiles against YCSB benchmarks, added DES engine profiling, and improved consolidated parquet ordering with union schema discovery.

### Saturday, Feb 28

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`d42bc1b`](https://github.com/cdouglas/endive/commit/d42bc1b) | feature | (moderate) | Sort consolidated parquet by swept params, union schema discovery |
| [`3bded46`](https://github.com/cdouglas/endive/commit/3bded46) | bugfix | (trivial) | Set TZ in dev container, not production Dockerfile |
| [`da06b55`](https://github.com/cdouglas/endive/commit/da06b55) | bugfix | (trivial) | Set container timezone to America/Los_Angeles |
| [`64bc839`](https://github.com/cdouglas/endive/commit/64bc839) | analysis | (moderate) | Verify provider latency profiles against source benchmarks |
| [`d6abe9a`](https://github.com/cdouglas/endive/commit/d6abe9a) | analysis | (moderate) | DES engine profiling report from 233 experiment runs |
| [`66715ce`](https://github.com/cdouglas/endive/commit/66715ce) | chore | (minor) | Clean up stale documentation, update SPEC.md to v2.1 |
| [`fbe153a`](https://github.com/cdouglas/endive/commit/fbe153a) | feature | (moderate) | Add DES engine profiling (--profile flag) |
| [`64ee8de`](https://github.com/cdouglas/endive/commit/64ee8de) | docs | (minor) | Add analysis docs informing latency choices |
| [`e4f6097`](https://github.com/cdouglas/endive/commit/e4f6097) | bugfix | (moderate) | Update S3 Express GET/PUT latencies from independent benchmark |
| [`f558e9a`](https://github.com/cdouglas/endive/commit/f558e9a) | bugfix | (moderate) | Update unconditional GET/PUT latencies from independent benchmarks |

### Friday, Feb 27

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`fde2d2d`](https://github.com/cdouglas/endive/commit/fde2d2d) | bugfix | (minor) | Add provider/table titles to exp4c latency plots, fix Azure Premium name |
| [`6183a71`](https://github.com/cdouglas/endive/commit/6183a71) | bugfix | (moderate) | Add storage_provider to heatmap param extraction, rework exp4c plots |

### Thursday, Feb 26

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`c4e4658`](https://github.com/cdouglas/endive/commit/c4e4658) | feature | (major) | Add exp4c — multi-table contention across real storage providers |
| [`2098750`](https://github.com/cdouglas/endive/commit/2098750) | bugfix | (trivial) | Increase latency table precision to 2 decimal places |
| [`d93c5dd`](https://github.com/cdouglas/endive/commit/d93c5dd) | bugfix | (minor) | Scale all plot output to 1200px wide (DPI 300 → 100) |
| [`2ffde1d`](https://github.com/cdouglas/endive/commit/2ffde1d) | bugfix | (minor) | Put inter_arrival_scale on x-axis for exp4a heatmaps |
| [`5b39e49`](https://github.com/cdouglas/endive/commit/5b39e49) | docs | (minor) | Add beads task tracking rules to CLAUDE.md |
| [`a8ad46e`](https://github.com/cdouglas/endive/commit/a8ad46e) | feature | (moderate) | Auto-emit companion .md tables for every plot |

### Wednesday, Feb 25

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`5f4563a`](https://github.com/cdouglas/endive/commit/5f4563a) | feature | (moderate) | Add hatched xheatmap variants flagging low success rate cells |
| [`a7db949`](https://github.com/cdouglas/endive/commit/a7db949) | docs | (moderate) | Rewrite exp4 report with post-CAS-fix results |
| [`dc731d1`](https://github.com/cdouglas/endive/commit/dc731d1) | feature | (trivial) | Add CAS latency to exp4a heatmap titles |
| [`5678dc2`](https://github.com/cdouglas/endive/commit/5678dc2) | feature | (minor) | Add title_suffix to heatmap plots, show CAS latency in exp4b titles |
| [`8fb5544`](https://github.com/cdouglas/endive/commit/8fb5544) | feature | (minor) | Swap exp4b heatmap axes to put inter-arrival on x, num_tables on y |
| [`7d0ff60`](https://github.com/cdouglas/endive/commit/7d0ff60) | bugfix | (major) | Filter exp4 plots by catalog latency to prevent cross-latency mixing |
| [`e7f3343`](https://github.com/cdouglas/endive/commit/e7f3343) | feature | (moderate) | Parallelize config processing in regenerate_plots.py |

### Tuesday, Feb 24

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`e56e7d8`](https://github.com/cdouglas/endive/commit/e56e7d8) | feature | (minor) | Add per-simulation progress tracking for Docker monitoring |
| [`fe617c2`](https://github.com/cdouglas/endive/commit/fe617c2) | docs | (minor) | Document global seq as single-file catalog contention model |
| [`bfc4652`](https://github.com/cdouglas/endive/commit/bfc4652) | feature | (major) | Add exp4 multi-table catalog contention experiments |
| [`a1c0c32`](https://github.com/cdouglas/endive/commit/a1c0c32) | bugfix | (minor) | Move success rate annotations to P50 line with overlap detection |
| [`aac275e`](https://github.com/cdouglas/endive/commit/aac275e) | feature | (minor) | Add `--destructive` option to consolidation script |
| [`cde12a5`](https://github.com/cdouglas/endive/commit/cde12a5) | bugfix | (minor) | Reject `--destructive` with `--verify`/`--verify-only` |
| [`88d0530`](https://github.com/cdouglas/endive/commit/88d0530) | docs | (minor) | Add exp4 multi-table catalog contention report |
| [`65379f1`](https://github.com/cdouglas/endive/commit/65379f1) | bugfix | (moderate) | Spec fix: cross-table CAS failures should not pay per-attempt I/O |
| [`7da2c84`](https://github.com/cdouglas/endive/commit/7da2c84) | bugfix | (major) | Cross-table CAS failures skip per-attempt I/O on retry |
| [`4e4b6d2`](https://github.com/cdouglas/endive/commit/4e4b6d2) | docs | (minor) | Align SPEC.md with cross-table retry implementation |
| [`6a0aceb`](https://github.com/cdouglas/endive/commit/6a0aceb) | feature | (moderate) | Rename consolidation script and add `--partition` mode |
| [`369ee28`](https://github.com/cdouglas/endive/commit/369ee28) | chore | (trivial) | Increase Docker timeout to 4h |
| [`3f34765`](https://github.com/cdouglas/endive/commit/3f34765) | feature | (moderate) | Discover experiments from consolidated.parquet when directories absent |

### Monday, Feb 23

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`7b2790c`](https://github.com/cdouglas/endive/commit/7b2790c) | bugfix | (moderate) | Pay validation IO before real conflict detection |
| [`9bac1ae`](https://github.com/cdouglas/endive/commit/9bac1ae) | bugfix | (minor) | Exclude analysis-only code from experiment hash |
| [`4e878da`](https://github.com/cdouglas/endive/commit/4e878da) | bugfix | (minor) | Replace colorblind-unfriendly palettes in plots |
| [`6f9792f`](https://github.com/cdouglas/endive/commit/6f9792f) | feature | (moderate) | Expand exp3 parameter sweeps and refine plot configs |
| [`6384969`](https://github.com/cdouglas/endive/commit/6384969) | feature | (moderate) | Improve plot quality across all experiments |
| [`c182475`](https://github.com/cdouglas/endive/commit/c182475) | bugfix | (minor) | Update consolidation script for post-rewrite schema |
| [`1def59f`](https://github.com/cdouglas/endive/commit/1def59f) | bugfix | (trivial) | Accept title kwarg in generate_latency_vs_throughput_table |
| [`26a46aa`](https://github.com/cdouglas/endive/commit/26a46aa) | bugfix | (trivial) | Update stale config path in test_latency_separation |
| [`732a631`](https://github.com/cdouglas/endive/commit/732a631) | feature | (major) | Per-operation-type plots and per-type heatmaps for mixed workloads |
| [`7d7546a`](https://github.com/cdouglas/endive/commit/7d7546a) | chore | (minor) | Delete 60 inactive experiment configs |
| [`be348a3`](https://github.com/cdouglas/endive/commit/be348a3) | feature | (major) | Unified experiment config + plotting infrastructure |
| [`41a12ac`](https://github.com/cdouglas/endive/commit/41a12ac) | bugfix | (major) | Streaming parquet export to fix OOM in parallel runs |
| [`35ae7bd`](https://github.com/cdouglas/endive/commit/35ae7bd) | refactor | (moderate) | Extract provider config from Python dict to TOML files |
| [`9ae133a`](https://github.com/cdouglas/endive/commit/9ae133a) | docs | (minor) | Update SPEC.md to reflect current modular architecture |
| [`ead7fc2`](https://github.com/cdouglas/endive/commit/ead7fc2) | bugfix | (trivial) | Match container user to host user for consistent absolute paths |
| [`d53c7ef`](https://github.com/cdouglas/endive/commit/d53c7ef) | bugfix | (trivial) | Resolve plugin paths inside dev container |
| [`0d78f5c`](https://github.com/cdouglas/endive/commit/0d78f5c) | refactor | (minor) | Remove write_metadata() - manifest writes are plain PUTs |
| [`2fcb492`](https://github.com/cdouglas/endive/commit/2fcb492) | bugfix | (moderate) | Use YCSB manifest write latencies instead of PUT model |
| [`462d46e`](https://github.com/cdouglas/endive/commit/462d46e) | bugfix | (major) | Add per-attempt storage I/O and separate catalog/storage configs |

### Sunday, Feb 22

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`25a9493`](https://github.com/cdouglas/endive/commit/25a9493) | chore | (trivial) | Set real_conflict_probability=0.0 in exp2/exp3b configs |
| [`38fe1e2`](https://github.com/cdouglas/endive/commit/38fe1e2) | docs | (moderate) | Add refactoring specification and documentation index |
| [`4f22e08`](https://github.com/cdouglas/endive/commit/4f22e08) | feature | (moderate) | Add operation_type/t_runtime columns and handle catalog.backend=service |
| [`f8d308c`](https://github.com/cdouglas/endive/commit/f8d308c) | refactor | (major) | Migration complete - replace old main.py with new modules |
| [`267a130`](https://github.com/cdouglas/endive/commit/267a130) | refactor | (moderate) | Consolidate old modules into _legacy.py, delete redundant tests |
| [`8836b05`](https://github.com/cdouglas/endive/commit/8836b05) | feature | (major) | Add TOML config loading per SPEC.md §7 |
| [`372e917`](https://github.com/cdouglas/endive/commit/372e917) | feature | (major) | Implement Simulation runner and Statistics collector per SPEC.md §6 |
| [`50812bf`](https://github.com/cdouglas/endive/commit/50812bf) | feature | (major) | Implement ConflictDetector per SPEC.md §5 |
| [`7ecac81`](https://github.com/cdouglas/endive/commit/7ecac81) | feature | (major) | Implement Workload generator per SPEC.md §4 |
| [`b763da0`](https://github.com/cdouglas/endive/commit/b763da0) | feature | (major) | Implement Transaction types per SPEC.md §3 |
| [`457b1bf`](https://github.com/cdouglas/endive/commit/457b1bf) | feature | (major) | Implement uniform Catalog interface per SPEC.md §2 |
| [`1d0b236`](https://github.com/cdouglas/endive/commit/1d0b236) | feature | (major) | Implement StorageProvider interface per SPEC.md §1 |

---

## Week of Feb 15-21

Prepared for the architectural rewrite. Feb 17-20 fixed CAS modeling to use message-passing with server-time state capture, removed the groups abstraction, added table-level CAS conflict detection, and extracted transaction/snapshot modules. Feb 15-16 added partition scaling experiments with O(N) metadata cost, extracted config.py and utils.py from main.py, handled nested TOML key substitution, and added partition documentation. Dev container and beads issue tracker also set up.

### Saturday, Feb 21

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`0466074`](https://github.com/cdouglas/endive/commit/0466074) | chore | (trivial) | Install beads issue tracker in dev container |

### Thursday, Feb 20

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`d999f9a`](https://github.com/cdouglas/endive/commit/d999f9a) | bugfix | (trivial) | Add aws alias and reset provider fixtures in tests |
| [`9ef9f43`](https://github.com/cdouglas/endive/commit/9ef9f43) | bugfix | (minor) | Use correct provider name azurex instead of azure_premium |
| [`33f8095`](https://github.com/cdouglas/endive/commit/33f8095) | bugfix | (minor) | Quote string values in TOML config parameter substitution |
| [`045146f`](https://github.com/cdouglas/endive/commit/045146f) | bugfix | (minor) | Extract fast_append_ratio from operation_types config |
| [`97ce9ac`](https://github.com/cdouglas/endive/commit/97ce9ac) | bugfix | (moderate) | CAS conflict detection compares against start version, not expected write |
| [`ccf08c1`](https://github.com/cdouglas/endive/commit/ccf08c1) | test | (moderate) | Add table-level CAS conflict detection tests |
| [`e6fc270`](https://github.com/cdouglas/endive/commit/e6fc270) | refactor | (major) | Remove groups abstraction, simplify conflict detection |

### Wednesday, Feb 19

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`9f9319a`](https://github.com/cdouglas/endive/commit/9f9319a) | docs | (minor) | Add catalog/storage provider separation design |
| [`3c2c2be`](https://github.com/cdouglas/endive/commit/3c2c2be) | feature | (moderate) | Integrate catalog/storage providers into main.py |
| [`6640fa0`](https://github.com/cdouglas/endive/commit/6640fa0) | test | (moderate) | Add provider tests and s3x throughput validation |
| [`70b98d3`](https://github.com/cdouglas/endive/commit/70b98d3) | docs | (trivial) | Add operation type experiments to README |
| [`08201e3`](https://github.com/cdouglas/endive/commit/08201e3) | feature | (minor) | Add analysis scripts for operation type experiments |
| [`c229215`](https://github.com/cdouglas/endive/commit/c229215) | bugfix | (minor) | Initialize global latency defaults for import resolution |
| [`f30eba2`](https://github.com/cdouglas/endive/commit/f30eba2) | feature | (minor) | Add experiment config templates for operation type studies |
| [`125b772`](https://github.com/cdouglas/endive/commit/125b772) | feature | (minor) | Add operation type experiment groups to runner |
| [`5dc483d`](https://github.com/cdouglas/endive/commit/5dc483d) | feature | (moderate) | Add provider-specific min_latency from YCSB benchmarks |
| [`51f498e`](https://github.com/cdouglas/endive/commit/51f498e) | bugfix | (moderate) | Ensure strict event ordering with MIN_TIME_DELTA |

### Tuesday, Feb 18

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`9fc2abd`](https://github.com/cdouglas/endive/commit/9fc2abd) | docs | (moderate) | Add simulator fidelity analysis and internals documentation |
| [`a0bfeec`](https://github.com/cdouglas/endive/commit/a0bfeec) | feature | (major) | Add partition-aware conflict resolution with operation types |
| [`e92e062`](https://github.com/cdouglas/endive/commit/e92e062) | feature | (major) | Model operation types with accurate conflict resolution |
| [`a6d495b`](https://github.com/cdouglas/endive/commit/a6d495b) | refactor | (moderate) | Extract Tier 1 modules (snapshot.py, transaction.py) |
| [`ae69ac3`](https://github.com/cdouglas/endive/commit/ae69ac3) | refactor | (moderate) | Partition state between Catalog and Transactions via snapshots |
| [`b68b4cd`](https://github.com/cdouglas/endive/commit/b68b4cd) | bugfix | (moderate) | Table mode v_dirty not updated after catalog read, causing excess retries |
| [`e3e7992`](https://github.com/cdouglas/endive/commit/e3e7992) | bugfix | (major) | Model CAS as message-passing with server-time state capture |

### Monday, Feb 17

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`52c08e8`](https://github.com/cdouglas/endive/commit/52c08e8) | bugfix | (moderate) | Add catalog read latency after conflict resolution |
| [`e12f90a`](https://github.com/cdouglas/endive/commit/e12f90a) | feature | (minor) | Add partition columns to experiment index CSV export |
| [`469d91c`](https://github.com/cdouglas/endive/commit/469d91c) | chore | (trivial) | Remove run_all_experiments.sh shell wrapper |
| [`ce3a2ea`](https://github.com/cdouglas/endive/commit/ce3a2ea) | refactor | (moderate) | Consolidate verification into consolidation script, remove obsolete scripts |
| [`334a5cf`](https://github.com/cdouglas/endive/commit/334a5cf) | bugfix | (moderate) | Partition mode reads ML history like Iceberg's validationHistory |

### Monday, Feb 16

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`c2b7806`](https://github.com/cdouglas/endive/commit/c2b7806) | bugfix | (minor) | Drop inter_arrival.scale=10ms from experiment sweeps |
| [`9891aee`](https://github.com/cdouglas/endive/commit/9891aee) | bugfix | (trivial) | Change partition validation error to warning for low partition counts |
| [`cd9d59a`](https://github.com/cdouglas/endive/commit/cd9d59a) | docs | (trivial) | Update errata with refactoring progress |
| [`a2fbd21`](https://github.com/cdouglas/endive/commit/a2fbd21) | refactor | (minor) | Update test imports to use config.py and utils.py directly |
| [`f05abe1`](https://github.com/cdouglas/endive/commit/f05abe1) | refactor | (moderate) | Extract utils.py module from main.py (Phase 2) |
| [`d5747b5`](https://github.com/cdouglas/endive/commit/d5747b5) | refactor | (moderate) | Extract config.py module from main.py (Phase 3) |
| [`ad40f4a`](https://github.com/cdouglas/endive/commit/ad40f4a) | test | (minor) | Add tests for nested TOML section key substitution |
| [`bd930d1`](https://github.com/cdouglas/endive/commit/bd930d1) | bugfix | (moderate) | Handle nested TOML keys in experiment parameter substitution |
| [`8ddce8e`](https://github.com/cdouglas/endive/commit/8ddce8e) | test | (moderate) | Add tests for partition metadata scaling and conflict resolution |
| [`e7c9879`](https://github.com/cdouglas/endive/commit/e7c9879) | feature | (major) | Add partition scaling experiments and O(N) metadata cost |

### Sunday, Feb 15

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`0cbc945`](https://github.com/cdouglas/endive/commit/0cbc945) | docs | (trivial) | Add partition group to RUNNING_EXPERIMENTS.md |
| [`7585a83`](https://github.com/cdouglas/endive/commit/7585a83) | docs | (minor) | Add partition-level modeling documentation |
| [`e5af86c`](https://github.com/cdouglas/endive/commit/e5af86c) | docs | (minor) | Add partition-level modeling to ARCHITECTURE.md and model.md |
| [`02705a8`](https://github.com/cdouglas/endive/commit/02705a8) | feature | (minor) | Add warnings for meaningless partition configurations |

---

## Week of Feb 8-14

Replaced constant-time latencies with lognormal distributions in a five-phase rollout on Feb 11: distributions, storage/catalog config separation, precedence tests, failure multipliers, and YCSB validation. Provider profiles calibrated to June 2025 YCSB benchmarks. Added partition-level modeling on Feb 14 (per-table partition counts, distributed manifest lists, nonce-based deterministic seeds, dev container). Corrected experiment factorial design with regression tests on Feb 12. Fixed false conflict resolution to include manifest list operations on Feb 13.

### Saturday, Feb 14

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`32d1634`](https://github.com/cdouglas/endive/commit/32d1634) | feature | (moderate) | Add blog post experiments with instant catalog + real storage |
| [`61bdbe4`](https://github.com/cdouglas/endive/commit/61bdbe4) | feature | (minor) | Add dev container for Claude Code development |
| [`253f18d`](https://github.com/cdouglas/endive/commit/253f18d) | bugfix | (trivial) | Remove bash function export from run_blog_experiments.sh |
| [`777101c`](https://github.com/cdouglas/endive/commit/777101c) | bugfix | (moderate) | Use nonce-based deterministic seeds for reentrance |
| [`488ae1a`](https://github.com/cdouglas/endive/commit/488ae1a) | docs | (trivial) | Update CLAUDE.md for consolidated experiment runner |
| [`bfb6081`](https://github.com/cdouglas/endive/commit/bfb6081) | refactor | (moderate) | Consolidate experiment scripts and rename blog_* to instant_* |
| [`9a50284`](https://github.com/cdouglas/endive/commit/9a50284) | refactor | (moderate) | Extend to per-table partition model |
| [`fc6c4ca`](https://github.com/cdouglas/endive/commit/fc6c4ca) | feature | (major) | Add partition-level modeling with distributed manifest lists |

### Friday, Feb 13

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`84f22ed`](https://github.com/cdouglas/endive/commit/84f22ed) | feature | (minor) | Add git SHA tracking for experiment reproducibility |
| [`c8b1029`](https://github.com/cdouglas/endive/commit/c8b1029) | bugfix | (major) | Correct false conflict resolution to include manifest list operations |

### Thursday, Feb 12

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`9cd017f`](https://github.com/cdouglas/endive/commit/9cd017f) | feature | (moderate) | Add sustainable throughput plot (99%+ success region) |
| [`3b0d42f`](https://github.com/cdouglas/endive/commit/3b0d42f) | bugfix | (minor) | Update consolidation script to match current experiment naming |
| [`055b913`](https://github.com/cdouglas/endive/commit/055b913) | test | (minor) | Update distribution conformance tests for new experiment naming |
| [`2ecfc10`](https://github.com/cdouglas/endive/commit/2ecfc10) | docs | (minor) | Update README and RUNNING_EXPERIMENTS for new experiment design |
| [`a85344d`](https://github.com/cdouglas/endive/commit/a85344d) | feature | (minor) | Add Azure Premium (azurex) experiment configs |
| [`30200c6`](https://github.com/cdouglas/endive/commit/30200c6) | bugfix | (moderate) | Correct experiment factorial design and add regression tests |

### Tuesday, Feb 11

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`bd76c41`](https://github.com/cdouglas/endive/commit/bd76c41) | chore | (trivial) | Standardize venv directory to .venv |
| [`7098694`](https://github.com/cdouglas/endive/commit/7098694) | bugfix | (trivial) | Remove invalid logging package from requirements |
| [`3b7e8c6`](https://github.com/cdouglas/endive/commit/3b7e8c6) | refactor | (trivial) | Rename experiment configs for clarity |
| [`63cb742`](https://github.com/cdouglas/endive/commit/63cb742) | feature | (moderate) | Add optimization experiment configs and unified runner |
| [`7d01f25`](https://github.com/cdouglas/endive/commit/7d01f25) | feature | (minor) | Add blog experiment runner script |
| [`35f184a`](https://github.com/cdouglas/endive/commit/35f184a) | refactor | (minor) | Archive old experiment configs and add new blog post configs |
| [`45252bf`](https://github.com/cdouglas/endive/commit/45252bf) | bugfix | (moderate) | Add table metadata read/write to conflict resolution |
| [`0bec573`](https://github.com/cdouglas/endive/commit/0bec573) | feature | (moderate) | Add size-based PUT latency model and clean up docs |
| [`c76847c`](https://github.com/cdouglas/endive/commit/c76847c) | bugfix | (minor) | Use num_groups=1 for catalog-level contention in exp8 configs |
| [`2be773c`](https://github.com/cdouglas/endive/commit/2be773c) | feature | (minor) | Add exp8 experiment runner and Azure standard configs |
| [`c14c010`](https://github.com/cdouglas/endive/commit/c14c010) | feature | (moderate) | Add separate failure distributions and experiment configs |
| [`8aa21d2`](https://github.com/cdouglas/endive/commit/8aa21d2) | test | (moderate) | Add comprehensive distribution validation tests |
| [`2bf7904`](https://github.com/cdouglas/endive/commit/2bf7904) | bugfix | (major) | Correct provider profiles to match June 2025 YCSB benchmarks |
| [`c840f33`](https://github.com/cdouglas/endive/commit/c840f33) | test | (moderate) | Add validation tests for realistic latency modeling (Phase 5) |
| [`0ece4ef`](https://github.com/cdouglas/endive/commit/0ece4ef) | feature | (major) | Add failure latency multiplier and contention scaling (Phase 4) |
| [`aa7ff2e`](https://github.com/cdouglas/endive/commit/aa7ff2e) | test | (moderate) | Add configuration precedence tests (Phase 3) |
| [`a1a9c7c`](https://github.com/cdouglas/endive/commit/a1a9c7c) | feature | (moderate) | Separate storage and catalog configuration (Phase 2) |
| [`16851eb`](https://github.com/cdouglas/endive/commit/16851eb) | feature | (major) | Add lognormal latency distributions (Phase 1) |
| [`485f235`](https://github.com/cdouglas/endive/commit/485f235) | docs | (minor) | Add realistic latency modeling plan based on cloud measurements |

---

## Week of Feb 1-7

Implemented manifest list append protocol (append entries instead of rewriting the full list) and table metadata inlining. The protocol went through several corrections: separating physical append from verification, adding catalog re-read costs. Added experiment configs for append operations and a `compaction_max_entries` parameter to bound manifest list growth.

### Wednesday, Feb 4

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`f73dcaa`](https://github.com/cdouglas/endive/commit/f73dcaa) | feature | (major) | Add table metadata inlining and manifest list append protocol |
| [`db4627a`](https://github.com/cdouglas/endive/commit/db4627a) | bugfix | (moderate) | Add catalog re-read cost after physical append success |
| [`ec170b4`](https://github.com/cdouglas/endive/commit/ec170b4) | refactor | (moderate) | Simplify append protocol - validation at append time |
| [`3314826`](https://github.com/cdouglas/endive/commit/3314826) | docs | (trivial) | Remove duplicate line in append simulation design |
| [`7d20182`](https://github.com/cdouglas/endive/commit/7d20182) | docs | (minor) | Update errata with protocol correction details |
| [`d89cfd2`](https://github.com/cdouglas/endive/commit/d89cfd2) | bugfix | (major) | Correct append protocol - separate physical append from verification |
| [`de95442`](https://github.com/cdouglas/endive/commit/de95442) | feature | (moderate) | Add compaction_max_entries parameter for entry-count-based compaction |
| [`e6d773a`](https://github.com/cdouglas/endive/commit/e6d773a) | docs | (moderate) | Add append simulation design document |

### Tuesday, Feb 3

| Commit | Type | Sig | Description |
|--------|------|-----|-------------|
| [`3c2bfd7`](https://github.com/cdouglas/endive/commit/3c2bfd7) | feature | (moderate) | Implement Phase 4 - Experiment configs for append operations |
| [`4919630`](https://github.com/cdouglas/endive/commit/4919630) | feature | (major) | Implement Phase 2 - Manifest List Append Mode |

---

## Legend

| Label | Meaning |
|-------|---------|
| (trivial) | Typo fix, config tweak, cleanup |
| (minor) | Small feature, doc update, simple bugfix |
| (moderate) | Meaningful feature or important bugfix |
| (major) | Architectural change, critical bugfix, or key new capability |
