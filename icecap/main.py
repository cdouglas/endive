#!/usr/bin/env python

import itertools
import logging
import simpy
import tomllib
import numpy as np
from capstats import Stats, truncated_zipf_pmf, lognormal_params_from_mean_and_sigma
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

N_TABLES: int
N_TXN_RETRY: int
T_CAS: int
T_METADATA_ROOT: int
T_MANIFEST_LIST: int
T_MANIFEST_FILE: int
# lognormal distribution of transaction runtimes
T_MIN_RUNTIME: int
T_RUNTIME_MU: float
T_RUNTIME_SIGMA: float
# exponential inter-arrival rate for transactions (ms; 1000 = ~ 1/sec)
T_TXN_INTER_ARRIVAL: int
# tables per transaction (prob. mass function)
N_TBL_PMF: float
# which tables are selected; (zipf, 0 most likely, so on)
TBL_R_PMF: float
# number of tables written (subset read)
N_TBL_W_PMF: float

STATS = Stats()

def configure_from_toml(config_file: str):
    global N_TABLES, T_CAS, T_METADATA_ROOT, T_MANIFEST_LIST, T_MANIFEST_FILE
    global T_MIN_RUNTIME, T_RUNTIME_MU, T_RUNTIME_SIGMA, T_TXN_INTER_ARRIVAL
    global N_TBL_PMF, TBL_R_PMF, N_TBL_W_PMF, N_TXN_RETRY

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    # Load basic integer configuration
    N_TABLES = config["catalog"]["num_tables"]

    # TODO distr around these
    T_CAS = config["storage"]["T_CAS"]
    T_METADATA_ROOT = config["storage"]["T_METADATA_ROOT"]
    T_MANIFEST_LIST = config["storage"]["T_MANIFEST_LIST"]
    T_MANIFEST_FILE = config["storage"]["T_MANIFEST_FILE"]

    # Load runtime-related configuration
    N_TXN_RETRY = config["transaction"]["retry"]
    T_MIN_RUNTIME = config["transaction"]["runtime"]["min"] #["T_MIN_RUNTIME_MS"]
    mean = config["transaction"]["runtime"]["mean"] #["T_RUNTIME_MEAN"]
    sigma = config["transaction"]["runtime"]["sigma"] #["T_RUNTIME_SIGMA"]
    T_RUNTIME_MU, T_RUNTIME_SIGMA = lognormal_params_from_mean_and_sigma(mean, sigma)
    # Load transaction inter-arrival time
    T_TXN_INTER_ARRIVAL = config["transaction"]["inter_arrival"] #["T_TXN_INTER_ARRIVAL_MS"]

    # Load parameters for PMFs
    ntbl_exponent = config.get("transaction", {}).get("ntable", {}).get("zipf", 2.0)
    tblr_exponent = config.get("transaction", {}).get("seltbl", {}).get("zipf", 1.4)
    ntblw_exponent = config.get("transaction", {}).get("seltblw", {}).get("zipf", 1.2)

    # Generate PMFs
    N_TBL_PMF = truncated_zipf_pmf(N_TABLES, ntbl_exponent)
    TBL_R_PMF = truncated_zipf_pmf(N_TABLES, tblr_exponent)
    N_TBL_W_PMF = [truncated_zipf_pmf(k, ntblw_exponent) for k in range(0, N_TABLES + 1)]

class Catalog:
    def __init__(self, sim):
        self.sim = sim
        self.seq = 0
        self.tbl = [0] * N_TABLES

    # this is too coarse-grained for append/etc. but revisit later
    def try_CAS(self, sim, txn):
        logger.debug(f"{sim.now} TXN {txn.id} CAS {self.seq} = {txn.v_catalog_seq} {txn.v_tblw}")
        logger.debug(f"{sim.now} TXN {txn.id} Catalog {self.tbl}")
        if self.seq == txn.v_catalog_seq:
            for off, val in txn.v_tblw.items():
                self.tbl[off] = val
            self.seq += 1
            logger.debug(f"{sim.now} TXN {txn.id} CASOK   {self.tbl}")
            return True
        return False

@dataclass
class Txn:
    id: int
    t_submit: int # ms submitted since start
    t_runtime: int # ms between submission and commit
    v_catalog_seq: int # version of catalog read (CAS in storage)
    v_tblr: dict[int, int] # versions of tables read
    v_tblw: dict[int, int] # versions of tables written
    n_retries: int # number of retries
    t_commit: int = field(default=-1)
    t_abort: int = field(default=-1)
    v_dirty: dict[int, int] = field(default_factory=lambda: defaultdict(dict)) # versions validated (init union(v_tblr, v_tblw))

def txn_ml_w(sim, txn):
    # write each manifest list TODO: not sequential?
    for _ in txn.v_tblw:
        yield sim.timeout(T_MANIFEST_LIST) # TODO: distr
    logger.debug(f"{sim.now} TXN {txn.id} ML_W")

def txn_commit(sim, txn, catalog):
    # TODO move commit to Catalog to test CASCatalog, AppendCatalog?
    yield sim.timeout(T_CAS) # CAS >
    if catalog.try_CAS(sim, txn): # txn.v_catalog_seq, txn.v_tblw): # success?
        logger.debug(f"{sim.now} TXN {txn.id} commit")
        txn.t_commit = sim.now # known committed at this tick
        STATS.commit(txn)
    else:
        logger.debug(f"{sim.now} TXN {txn.id} CAS Fail")
        if txn.n_retries > N_TXN_RETRY:
            txn.t_abort = sim.now
            STATS.abort(txn)
            return
        yield sim.timeout(T_CAS / 2) # CAS > failed, read catalog
        # record catalog sequence number and versions read
        v_catalog = dict()
        txn.v_catalog_seq = catalog.seq
        for t in txn.v_dirty.keys():
            v_catalog[t] = catalog.tbl[t]
        yield sim.timeout(T_CAS / 2) # < CAS

        for t, v in txn.v_dirty.items():
            # optimistic, parallel version
            if not v_catalog[t] == v:
                # TODO add noise to reads/writes (even in parallel, typ max)
                # read
                yield sim.timeout(T_METADATA_ROOT)
                yield sim.timeout(T_MANIFEST_LIST)
                # TODO skip T_MANIFEST_FILE with some prob
                yield sim.timeout(T_MANIFEST_FILE) # min height of tree
                # write updated (merged) metadata
                yield sim.timeout(T_MANIFEST_FILE)
                yield sim.timeout(T_MANIFEST_LIST)

                # update validation to current
                txn.v_dirty[t] = v_catalog[t]
        # update write set to the next available version per table (confluent w)
        for t in txn.v_tblw.keys():
            txn.v_tblw[t] = v_catalog[t] + 1


def rand_tbl(catalog):
    # how many tables
    ntbl = int(np.random.choice(np.arange(1, N_TABLES + 1), p=N_TBL_PMF))
    # which tables read
    tblr_idx = np.random.choice(np.arange(0, N_TABLES), size=ntbl, replace=False, p=TBL_R_PMF).astype(int).tolist()
    tblr = {t: catalog.tbl[t] for t in tblr_idx}
    tblr_idx.sort()
    # write \subseteq read (not empty, read-only txn snapshot)
    ntblw = int(np.random.choice(np.arange(1, ntbl + 1), p=N_TBL_W_PMF[ntbl]))
    # uniform random from #tables to write
    tblw_idx = np.random.choice(tblr_idx, size=ntblw, replace=False).astype(int).tolist()
    # write versions = catalog versions + 1
    tblw = {t: tblr[t] + 1 for t in tblw_idx}
    return tblr, tblw

def txn_gen(sim, txn_id, catalog):
    tblr, tblw = rand_tbl(catalog)
    t_runtime = T_MIN_RUNTIME + np.random.lognormal(mean=T_RUNTIME_MU, sigma=T_RUNTIME_SIGMA)
    logger.debug(f"{sim.now} TXN {txn_id} {t_runtime} r {tblr} w {tblw}")
    # check all versions read/written TODO: serializable vs snapshot
    txn = Txn(txn_id, sim.now, int(t_runtime), catalog.seq, tblr, tblw)
    txn.v_dirty = txn.v_tblr.copy()
    txn.v_dirty.update(txn.v_tblw)
    # run the transaction
    yield sim.timeout(txn.t_runtime)
    # write the manifest list
    yield sim.process(txn_ml_w(sim, txn))
    while txn.t_commit < 0 and txn.t_abort < 0:
        # attempt commit
        txn.n_retries += 1
        yield sim.process(txn_commit(sim, txn, catalog))

def setup(sim): # TODO rand seed
    catalog = Catalog(sim)
    txn_ids = itertools.count(1)
    sim.process(txn_gen(sim, next(txn_ids), catalog))
    while True:
        yield sim.timeout(int(np.random.exponential(scale=T_TXN_INTER_ARRIVAL)))
        sim.process(txn_gen(sim, next(txn_ids), catalog))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(filename='est.log', level=logging.DEBUG)
    configure_from_toml("cfg.toml")

    # detn replay
    seed = np.random.randint(0, 2**32 -1)
    np.random.seed(seed)
    logger.info(f"SEED: {seed}")

    env = simpy.Environment()
    env.process(setup(env))
    env.run(until=100000000)
