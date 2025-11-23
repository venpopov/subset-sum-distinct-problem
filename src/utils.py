"""
Utility functions shared across scripts.
"""

from typing import List, Tuple


def P_from_d_prefix(d_prefix: List[int]) -> List[int]:
    """
    Given a prefix d_1..d_n, construct the corresponding decreasing P:

        p_k = sum_{j=k}^n d_j

    Returns:
        P_n = [p_1, ..., p_n] with p_1 > ... > p_n > 0.
    """
    s = 0
    p_rev = []
    for dj in reversed(d_prefix):
        s += dj
        p_rev.append(s)
    return list(reversed(p_rev))


def parse_seed(seed_str: str) -> Tuple[int, int]:
    """
    Parse a seed string of the form 'a,b' into a tuple (a, b).

    Example:
        '2,1' -> (2, 1)
    """
    parts = seed_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid seed format: {seed_str!r}, expected 'a,b'")
    try:
        a = int(parts[0].strip())
        b = int(parts[1].strip())
    except ValueError:
        raise ValueError(f"Seed components must be integers: {seed_str!r}")
    return a, b


def parse_seed_list(seed_str: str) -> List[int]:
    """
    Parse a seed string of the form 'a,b,...' into a list [a, b, ...].

    Example:
        '2,1' -> [2, 1]
        '1,2,3' -> [1, 2, 3]
    """
    parts = seed_str.split(",")
    if len(parts) < 2:
        raise ValueError(f"Invalid seed format: {seed_str!r}, expected at least 2 comma-separated integers")
    try:
        return [int(part.strip()) for part in parts]
    except ValueError:
        raise ValueError(f"Seed components must be integers: {seed_str!r}")
