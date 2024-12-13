#!/usr/bin/env python3

import simpy
import heapq
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, labs, theme_minimal
from dataclasses import dataclass, field

class Catalog:
    def __init__(self, env, ntables):
        self.env = env
        self.seq = 0
        self.tbl = [0] * ntables

    def try_CAS(self, seq, wtbl):
        if self.seq == seq:
            for off, val in wtbl.items():
                self.tbl[off] = val
            self.seq += 1
            return True
        return False

@dataclass
class Txn:
    nextt: int
    initt: int = field(compare=False)
    tblr: list = field(default_factory=list, compare=False)
    tblw: list = field(default_factory=list, compare=False)

def cas_time(rtt, n, p, delayf):
    """
    Calculate the expected transaction time with configurable delay

    Parameters:
        rtt (float): Round trip time in milliseconds.
        n (int): Maximum number of retries.
        p (float): Probability of success for compare-and-swap.
        delayf: e.g., exponential backoff

    Returns:
        float: Expected transaction time in milliseconds.
    """
    # Ensure inputs are valid
    if not (0 < p <= 1):
        raise ValueError("Probability of success (p) must be in the range (0, 1].")
    if n < 1:
        raise ValueError("Number of retries (n) must be at least 1.")

    # Calculate the summation S_n for limited retries
    one_minus_p = 1 - p
    sum_sn = 0

    for k in range(1, n + 1):
        delay = next(delayf) # 2**(k - 1)  # Exponential backoff delay multiplier
        sum_sn += one_minus_p**k * delay

    # Final S_n value
    sn = one_minus_p * sum_sn

    # Calculate T_retry
    t_retry = rtt * (1 + sn)

    # Calculate T_0
    t_0 = rtt * (p + one_minus_p * (1 + t_retry / rtt))

    return t_0

def conflict_time(rtt, n, p, c):
    """
    Calculate time to 
    """
    pass

def multi_iter(i, j, k):
    try:
        while True:
            yield(next(i), next(j), next(k))
    except StopIteration:
        return

# Function to explore a range of n values and generate a plot
def explore_and_plot(rtti, ni, pi, delayf):
    """
    Explore a range of n values and generate a plot of expected transaction time.

    Parameters:
        rtt (int): Round trip time in milliseconds.
        max_n (int): Maximum number of retries to explore.
        p (float): Probability of success for compare-and-swap.

    Returns:
        None
    """
    # Create a DataFrame to store results
    results = []

    for (rtt, n, p) in multi_iter(rtti, ni, pi): # range(1, max_n + 1):
        expected_time = cas_time(rtt, n, p, delayf)
        # results.append({"RTT": rtt, "ExpectedTime": expected_time})
        results.append({"Retries": n, "ExpectedTime": expected_time})
        # results.append({"Prob success": p, "ExpectedTime": expected_time})

    df = pd.DataFrame(results)

    # Generate the plot using ggplot
    plot = (
        ggplot(df, aes(x="Retries", y="ExpectedTime")) +
        geom_line() +
        labs(
            title="Expected Transaction Time vs Number of Retries",
            x="Number of Retries (n)",
            y="Expected Transaction Time (ms)"
        ) +
        theme_minimal()
    )

    print(plot)
    plot.save("rtt.png")

if __name__ == "__main__":
    rtt = itertools.repeat(100) # fix rtt to 100ms
    n = iter(range(1,10))       # 1..10 retries
    p = itertools.repeat(0.80)  # 80% chance of successful CAS
    delayf = (min(100 * 2**(k - 1), 1800000) for k in itertools.count(1)) # BaseTransaction:L356

    try:
        explore_and_plot(rtt, n, p, delayf)
    except ValueError as e:
        print(f"Error: {e}")

