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
