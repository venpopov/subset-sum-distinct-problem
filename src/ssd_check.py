"""
Independent brute-force checkers for the Distinct Subset Sums (DSS) property.

We provide two different implementations:

1. combinations-based: uses itertools.combinations over all subset sizes
2. bitmask-based: uses integer masks 0..2^n - 1
3. bitsets-based

Both return:
    (bool, info)

- bool: True if all subset sums are distinct, False if a collision is found.
- info: a message or a tuple describing the colliding subsets.
"""

from itertools import combinations
from typing import List, Tuple, Dict, Union


Subset = Tuple[int, ...]
CheckResult = Tuple[bool, Union[str, Tuple[Subset, Subset, int]]]


def is_distinct_subset_sum_combinations(nums: List[int]) -> CheckResult:
    """
    Brute-force DSS check using combinations over all subset sizes.

    Args:
        nums: list of positive integers.

    Returns:
        (True, "All subset sums are distinct.") if DSS,
        (False, (subset1, subset2, s)) if a collision with sum s is found.
    """
    seen_sums: Dict[int, Subset] = {}
    n = len(nums)

    for r in range(n + 1):
        for subset in combinations(nums, r):
            s = sum(subset)
            if s in seen_sums:
                return False, (seen_sums[s], subset, s)
            seen_sums[s] = subset

    return True, "All subset sums are distinct."


def is_distinct_subset_sum_bitmask(nums: List[int]) -> CheckResult:
    """
    Brute-force DSS check using bitmask enumeration of all subsets.

    Args:
        nums: list of positive integers.

    Returns:
        (True, "All subset sums are distinct.") if DSS,
        (False, (subset1, subset2, s)) if a collision with sum s is found.
    """
    n = len(nums)
    seen_sums: Dict[int, Subset] = {}

    for mask in range(1 << n):
        s = 0
        subset = []
        m = mask
        i = 0
        while m:
            if m & 1:
                s += nums[i]
                subset.append(nums[i])
            i += 1
            m >>= 1
        subset_t = tuple(subset)
        if s in seen_sums:
            return False, (seen_sums[s], subset_t, s)
        seen_sums[s] = subset_t

    return True, "All subset sums are distinct."

def is_distinct_subset_sum_optimized(nums: List[int]) -> Tuple[bool, str]:
    """
    Optimized DSS check using bitsets.
    Complexity: O(n * sum(P) / 64) instead of O(2^n).
    """
    # Bitmask representing {0}.
    # Bit i is 1 if sum 'i' is possible.
    reachable = 1

    total_sum = 0

    for x in nums:
        # Shift reachable left by x to get all new sums formed by adding x.
        # If (reachable & shifted) is non-zero, it means a sum in the 'new' set
        # overlaps with a sum in the 'old' set. Collision!
        if reachable & (reachable << x):
            return False, f"Collision detected involving {x}"

        # Merge the new sums into the reachable set
        reachable |= reachable << x
        total_sum += x

    return True, "All subset sums are distinct."


if __name__ == "__main__":
    # Small self-check / demo
    test_sets = {
        "GC_n5": [13, 12, 11, 9, 6],  # known DSS
        "simple_bad": [3, 4, 5, 6],  # 3+6 = 4+5
    }

    for name, nums in test_sets.items():
        ok_c, info_c = is_distinct_subset_sum_combinations(nums)
        ok_b, info_b = is_distinct_subset_sum_bitmask(nums)
        ok_o, info_o = is_distinct_subset_sum_optimized(nums)
        print(f"{name}: comb_ok={ok_c}, mask_ok={ok_b}, opt_ok={ok_o}")
        if not ok_c:
            print("  combinations collision:", info_c)
        if not ok_b:
            print("  bitmask collision:", info_b)
        if not ok_o:
            print("  bitsets collision:", info_o)
