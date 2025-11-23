import time
import gc
from typing import List, Dict


def optimize_motzkin_greedy(max_n: int, d_initial: List[int]) -> List[int]:
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

    # Track the current step index (0-based)
    current_step = 0

    # Helper to advance the DP state by one step using a specific weight
    def advance_layers(current_layers: Dict[int, int], weight: int, step_index: int) -> Dict[int, int]:
        new_layers: Dict[int, int] = {}

        # OPTIMIZATION 1: Geometric Pruning
        # The absolute last step we care about is (max_n - 2).
        # Any path must be able to return to height 0 by then.
        # So, at the *next* step (step_index + 1), max valid height is:
        steps_remaining = (max_n - 1) - (step_index + 1)
        max_allowed_height = steps_remaining

        # OPTIMIZATION 2: Eager Memory Release
        # We convert keys to a list so we can modify the dict (pop) while iterating.
        # Sorting isn't strictly necessary but helps visualize processing order.
        active_heights = sorted(list(current_layers.keys()))

        for h in active_heights:
            # destructive read: removes the massive integer from memory immediately
            mask = current_layers.pop(h)

            # Try all valid Motzkin moves: delta in {-1, 0, +1}
            for delta in (-1, 0, 1):
                nh = h + delta
                if nh < 0:
                    continue

                if nh > max_allowed_height:
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

            # Hint to Python allocator (optional, but good for massive objects)
            del mask

        return new_layers

    # --- INITIALIZATION ---
    # We need to advance the state through all initial values
    # The loop will start at n = len(d_initial) + 1
    # For that iteration, we need the state at step n-2 = len(d_initial) - 1
    # So we advance through d[1], d[2], ..., d[len(d_initial) - 1]
    for i in range(1, len(d_initial)):
        layers = advance_layers(layers, d[i], current_step)
        current_step += 1

    # --- MAIN LOOP ---
    start_n = len(d_initial) + 1
    for n in range(start_n, max_n + 1):
        # 1. IDENTIFY FORBIDDEN SUMS
        # To close the path at step n-1 (return to height 0),
        # we must currently be at height 0 or height 1 (since max drop is 1).
        # We combine these masks to get all "dangerous" sums.
        forbidden_mask = layers.get(0, 0) | layers.get(1, 0)

        # 2. Find smallest missing positive integer (Gap Trick)
        if forbidden_mask == 0:
            k = 1  # Should not happen for standard seeds, but safe fallback
        else:
            m = forbidden_mask >> 1
            lowest_zero_bit = (~m) & (m + 1)
            k = lowest_zero_bit.bit_length()
            # Edge case: if mask was 111...111 with no zeros, k is len.
            # Realistically, the mask is sparse-ish, so this works.
            if k == 0:
                k = 1  # If bit_length is 0 (meaning m was -1??), fallback.

        d.append(k)

        # 3. ADVANCE STATE
        # Prepare for the next n (which will need state at n-2).
        # We just finished n, so we advance using d[n-1] to get to step n-1.
        # Note: d is 1-based, so d[n-1] is the weight we just "passed".
        if n < max_n:
            weight_to_add = d[n - 1]  # Actually d[2] for n=3 loop preparing for n=4
            layers = advance_layers(layers, weight_to_add, current_step)
            current_step += 1

    return d[1:]


if __name__ == "__main__":
    # Test with seed (2, 1)
    TARGET_N = 38
    D_INITIAL = [1, 1]

    print(f"Calculating Motzkin-greedy sequence for seed {D_INITIAL} up to n={TARGET_N}...")

    start_time = time.perf_counter()
    result = optimize_motzkin_greedy(TARGET_N, D_INITIAL)
    end_time = time.perf_counter()

    print(f"\nResult: {result}")
    print(f"Time:   {(end_time - start_time):.6f} seconds")
