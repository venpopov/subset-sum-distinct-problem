import time
from typing import List, Dict


def motzkin_greedy_optimized(max_n: int, d1: int, d2: int) -> List[int]:
    """
    Computes the Motzkin-greedy sequence using efficient Bitset DP.

    Optimizations:
    1. Bitsets: Uses Python integers to represent sets of sums.
       (e.g., bit 5 is '1' -> sum 5 is reachable).
    2. Layer-by-Layer: Merges all paths ending at the same height/sum.
    3. Incremental: Never re-computes early layers. State persists across n.
    """
    # d is 1-based: d[1]=d1, d[2]=d2
    d = [0, d1, d2]

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
    # We start the loop at n=3, which requires the state at step n-2 = 1.
    # So we must manually advance the state past d[1] first.
    layers = advance_layers(layers, d[1])

    # --- MAIN LOOP ---
    for n in range(3, max_n + 1):
        # 1. IDENTIFY FORBIDDEN SUMS
        # To close the path at step n-1 (return to height 0),
        # we must currently be at height 0 or height 1 (since max drop is 1).
        # We combine these masks to get all "dangerous" sums.
        forbidden_mask = layers.get(0, 0) | layers.get(1, 0)

        # 2. FIND SMALLEST MISSING POSITIVE INTEGER
        # We look for the first '0' bit starting from index 1.
        k = 1
        while (forbidden_mask >> k) & 1:
            k += 1
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
    # Benchmark Comparison
    TARGET_N = 25
    D1, D2 = 2, 1

    print(f"Calculating Motzkin-greedy sequence for seed ({D1},{D2}) up to n={TARGET_N}...")

    start_time = time.perf_counter()
    result = motzkin_greedy_optimized(TARGET_N, D1, D2)
    end_time = time.perf_counter()

    print(f"\nResult: {result}")
    print(f"Time:   {(end_time - start_time):.6f} seconds")

    # Verification against your provided file data (first few terms of seed 1,1)
    # Expected (from uploaded summary_seed_1_1.csv):
    # 1, 1, 2, 3, 6, 11, 20, 40, 77, 148, 285, 570...
    expected_start = [1, 1, 2, 3, 6, 11, 20, 40, 77, 148]
    assert result[:10] == expected_start
    print("\n✓ Verification successful: Matches known Conway-Guy prefix.")
