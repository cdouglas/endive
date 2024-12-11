# https://chatgpt.com/share/6736919d-3324-8004-9802-0142f0fdaf88


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

# Input values
if __name__ == "__main__":
    print("Expected Transaction Time Calculator")
    rtt = 100.0 # float(input("Enter round trip time (RTT) in ms: "))
    n = 3 # int(input("Enter maximum number of retries (n): "))
    p = 0.8 #float(input("Enter probability of success (p): "))

    # Compute expected time
    try:
        expected_time = calculate_expected_time(rtt, n, p)
        print(f"\nExpected transaction time: {expected_time:.2f} ms")
    except ValueError as e:
        print(f"Error: {e}")

