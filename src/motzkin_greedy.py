import time
from typing import List, Dict


def motzkin_greedy_optimized(max_n: int, d_initial: List[int]) -> List[int]:
    """
    Computes the Motzkin-greedy sequence using efficient Bitset DP.

    Args:
        max_n: Maximum index n for the difference sequence to generate.
        d_initial: Initial values for the sequence (at least 2 elements).
                   These become d[1], d[2], d[3], ... d[len(d_initial)].

    Returns:
        Complete sequence d[1] through d[max_n].

    Optimizations:
    1. Bitsets: Uses Python integers to represent sets of sums.
       (e.g., bit 5 is '1' -> sum 5 is reachable).
    2. Layer-by-Layer: Merges all paths ending at the same height/sum.
    3. Incremental: Never re-computes early layers. State persists across n.
    """
    if len(d_initial) < 2:
        raise ValueError("d_initial must contain at least 2 elements")

    if max_n < len(d_initial):
        return d_initial[:max_n]

    # d is 1-based: d[0] is unused, d[1]=d_initial[0], d[2]=d_initial[1], etc.
    d = [0] + list(d_initial)

    # DP State: layers[h] = bitmask of all reachable sums ending at height h
    # Initial state: Step 0, Height 0, Sum 0 (2^0 = 1)
    layers: Dict[int, int] = {0: 1}

    # Helper to advance the DP state by one step using a specific weight
    def advance_layers(current_layers: Dict[int, int], weight: int) -> Dict[int, int]:
        new_layers: Dict[int, int] = {}
        for h, mask in current_layers.items():
            # Try all valid Motzkin moves: delta in {-1, 0, +1}
            for delta in (-1, 0, 1):
                nh = h + delta
                if nh < 0:
                    continue

                # The area added by this step is (new_height * weight)
                # We shift the bitmask left by this amount to add to all sums
                shift_amount = nh * weight
                shifted_mask = mask << shift_amount

                # Union with existing paths arriving at nh
                if nh in new_layers:
                    new_layers[nh] |= shifted_mask
                else:
                    new_layers[nh] = shifted_mask
        return new_layers

    # --- INITIALIZATION ---
    # We need to advance the state through all initial values
    # The loop will start at n = len(d_initial) + 1
    # For that iteration, we need the state at step n-2 = len(d_initial) - 1
    # So we advance through d[1], d[2], ..., d[len(d_initial) - 1]
    for i in range(1, len(d_initial)):
        layers = advance_layers(layers, d[i])

    # --- MAIN LOOP ---
    start_n = len(d_initial) + 1
    for n in range(start_n, max_n + 1):
        # 1. IDENTIFY FORBIDDEN SUMS
        # To close the path at step n-1 (return to height 0),
        # we must currently be at height 0 or height 1 (since max drop is 1).
        # We combine these masks to get all "dangerous" sums.
        forbidden_mask = layers.get(0, 0) | layers.get(1, 0)

        # 2. FIND SMALLEST MISSING POSITIVE INTEGER (Optimized)
        # We want the index of the first '0' bit in forbidden_mask, starting at bit 1.
        # Shift right to align bit 1 to bit 0
        m = forbidden_mask >> 1

        # Formula to isolate the lowest '0' bit:
        # 1. ~m turns the 0 we want into a 1.
        # 2. m + 1 creates a carry that ripples up to the first 0.
        # The intersection gives us exactly the bit we are looking for.
        lowest_zero_bit = (~m) & (m + 1)

        # The index is the length of the binary representation minus 1 (since it's a power of 2)
        # We add 1 because we shifted the mask initially.
        k = lowest_zero_bit.bit_length()

        d.append(k)

        # 3. ADVANCE STATE
        # Prepare for the next n (which will need state at n-2).
        # We just finished n, so we advance using d[n-1] to get to step n-1.
        # Note: d is 1-based, so d[n-1] is the weight we just "passed".
        if n < max_n:
            weight_to_add = d[n - 1]  # Actually d[2] for n=3 loop preparing for n=4
            layers = advance_layers(layers, weight_to_add)

    return d[1:]


if __name__ == "__main__":
    # Test with seed (2, 1)
    TARGET_N = 30
    D_INITIAL = [2, 1]

    print(f"Calculating Motzkin-greedy sequence for seed {D_INITIAL} up to n={TARGET_N}...")

    start_time = time.perf_counter()
    result = motzkin_greedy_optimized(TARGET_N, D_INITIAL)
    end_time = time.perf_counter()

    print(f"\nResult: {result}")
    print(f"Time:   {(end_time - start_time):.6f} seconds")

    # Verification against Conway-Guy (seed 1,1)
    # Expected (from uploaded summary_seed_1_1.csv):
    # 1, 1, 2, 3, 6, 11, 20, 40, 77, 148, 285, 570...
    print("\n--- Testing seed (1,1) for Conway-Guy verification ---")
    expected_start = [1, 1, 2, 3, 6, 11, 20, 40, 77, 148]
    result_conway = motzkin_greedy_optimized(10, [1, 1])
    assert result_conway == expected_start
    print("✓ Verification successful: Matches known Conway-Guy prefix.")

    # Test with longer initial sequence
    print("\n--- Testing with longer initial sequence [1, 2, 3] ---")
    result_long = motzkin_greedy_optimized(10, [1, 2, 3])
    print(f"Result: {result_long}")
