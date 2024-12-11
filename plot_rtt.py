#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, labs, theme_minimal

def calculate_expected_time(rtt, n, p):
    """
    Calculate the expected transaction time with exponential backoff.

    Parameters:
        rtt (float): Round trip time in milliseconds.
        n (int): Maximum number of retries.
        p (float): Probability of success for compare-and-swap.

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
        delay = 2**(k - 1)  # Exponential backoff delay multiplier
        sum_sn += one_minus_p**k * delay

    # Final S_n value
    sn = one_minus_p * sum_sn

    # Calculate T_retry
    t_retry = rtt * (1 + sn)

    # Calculate T_0
    t_0 = rtt * (p + one_minus_p * (1 + t_retry / rtt))

    return t_0

# Function to explore a range of n values and generate a plot
def explore_and_plot(rtt, max_n, p):
    """
    Explore a range of n values and generate a plot of expected transaction time.

    Parameters:
        rtt (float): Round trip time in milliseconds.
        max_n (int): Maximum number of retries to explore.
        p (float): Probability of success for compare-and-swap.

    Returns:
        None
    """
    # Create a DataFrame to store results
    results = []

    for n in range(1, max_n + 1):
        expected_time = calculate_expected_time(rtt, n, p)
        results.append({"Retries": n, "ExpectedTime": expected_time})

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

# Input values
if __name__ == "__main__":
    print("Expected Transaction Time Explorer")
    rtt = 100 # float(input("Enter round trip time (RTT) in ms: "))
    max_n = 10 # int(input("Enter maximum number of retries to explore: "))
    p = 0.8 # float(input("Enter probability of success (p): "))

    try:
        explore_and_plot(rtt, max_n, p)
    except ValueError as e:
        print(f"Error: {e}")

