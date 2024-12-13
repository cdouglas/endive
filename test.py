#!/usr/bin/env python

import itertools
import logging
import simpy
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

def chk_delay(k, delayf):
    for n in range(1, k):
        print(next(delayf))

def multi_iter(i, j, k):
    try:
        while True:
            yield(next(i), next(j), next(k))
    except StopIteration:
        return

def chk_multi_iter(i, j, k):
    for (ii, jj, kk) in multi_iter(i, j, k):
        print(ii, jj, kk)

class Catalog:
    def __init__(self, env, ntables):
        self.env = env
        self.seq = 0
        self.tbl = [0] * ntables

    # this is too coarse-grained for append/etc. but revisit later
    def try_CAS(self, seq, wtbl):
        logger.debug(f"CAS: {self.seq} {self.tbl}")
        if self.seq == seq:
            for off, val in wtbl.items():
                self.tbl[off] = val
            self.seq += 1
            return True
        return False

@dataclass
class Txn:
    id: int
    t_submit: int # = field(compare=False)
    t_runtime: int # ms running
    v_catalog_seq: int # version of catalog read (UUID in Iceberg)
    v_tblr: dict[int, int] # versions of tables read
    v_tblw: dict[int, int] # versions of tables written
    v_dirty: dict[int, int] = field(default_factory=lambda: defaultdict(dict)) # versions validated (init union(v_tblr, v_tblw))

# TODO store success/failure for transactions
class Stats:
    def __init__(self):
        pass

T_CAS = 100
T_METADATA_ROOT = 100
T_MANIFEST_LIST = 100
T_MANIFEST_FILE = 100

def txn_ml_w(sim, txn):
    # check all versions read/written TODO: serializable vs snapshot
    txn.v_dirty = txn.v_tblr.copy()
    txn.v_dirty.update(txn.v_tblw)
    # write each manifest list
    for _ in txn.v_tblw:
        yield sim.timeout(T_MANIFEST_LIST) # TODO: distr
    logger.debug(f"TXN {txn.id} ML_W {sim.now}")

def txn_commit(sim, txn, catalog):
    # TODO move commit to Catalog to test CASCatalog, AppendCatalog?
    yield sim.timeout(T_CAS) # attempt CAS
    if catalog.try_CAS(txn.v_catalog_seq, txn.v_tblw): # success?
        logger.debug(f"TXN {txn.id} commit   {sim.now}")
        txn.t_commit = sim.now # committed at this tick
    else:
        logger.debug(f"TXN {txn.id} CAS Fail {sim.now}")
        yield sim.timeout(T_CAS) # CAS failed, read catalog
        v_catalog = dict()
        # save catalog versions before yield
        for t in txn.v_dirty.keys():
            v_catalog[t] = catalog.tbl[t]
        for t, v in txn.v_dirty.items():
            # optimistic, parallel version
            if not v_catalog[t] == v:
                # TODO skip T_MANIFEST_FILE with some prob
                # TODO add noise to reads/writes (even in parallel, typ max)
                # read
                yield sim.timeout(T_METADATA_ROOT)
                yield sim.timeout(T_MANIFEST_LIST)
                yield sim.timeout(T_MANIFEST_FILE) # min height of tree
                # write updated (merged) metadata
                yield sim.timeout(T_MANIFEST_FILE)
                yield sim.timeout(T_MANIFEST_LIST)
            # pessimistic, serial version
            #   yield sim.timeout(T_METADATA_ROOT) # read root
            #   for _ in (catalog.tbl[t] - v):
            #       yield sim.timeout(T_MANIFEST_LIST) # each snapshot
            #       yield sim.timeout(T_MANIFEST_FILE) # TODO: distr

def txn_gen(sim, txn_id, catalog):
    tblr = {0: catalog.tbl[0], 3: catalog.tbl[3]} # TODO: distr
    tblw = dict()
    for t, v in tblr.items():
        tblw[t] = v + 1
    txn = Txn(txn_id, sim.now, 10000, catalog.seq, tblr, tblw)
    logger.debug(f"TXN {txn_id} init {sim.now}")
    # run the transaction
    yield sim.timeout(txn.t_runtime)
    # write the manifest list
    yield sim.process(txn_ml_w(sim, txn))
    # attempt commit
    yield sim.process(txn_commit(sim, txn, catalog))

def setup(sim, ntables): # TODO rand seed
    logger.debug("DEBUG0")
    catalog = Catalog(sim, ntables)
    txn_ids = itertools.count(1)
    sim.process(txn_gen(sim, next(txn_ids), catalog))
    while True:
        yield sim.timeout(1000) # TODO distr
        sim.process(txn_gen(sim, next(txn_ids), catalog))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(filename='est', level=logging.DEBUG)
    chk_delay(10, (2 * x for x in itertools.count()))
    chk_multi_iter(iter(range(1,5)), iter(range(1,3)), iter(range(1,3)))
    c = Catalog(None, 3)
    print(c.try_CAS(0, { 0: 1, 2: 1 }))
    print(c.try_CAS(0, { 0: 1, 2: 1 }))
    print(c.try_CAS(1, { 0: 2, 1: 1 }))

    logger.info("Starting sim")
    env = simpy.Environment()
    env.process(setup(env, 10))
    env.run(until=100000000)
