import time
import gc
from typing import List, Dict


def optimize_smooth_walk_greedy(max_n: int, d_initial: List[int]) -> List[int]:
    """
    Generates a difference sequence d(n) based on "Smooth Walk" constraints.

    Args:
        max_n: Maximum index n for the difference sequence to generate.
        d_initial: Initial values for the sequence (at least 2 elements).
                   These become d[1], d[2], d[3], ... d[len(d_initial)].

    Rules:
    - Walks start at 0. Steps in {-1, 0, 1}.
    - Valid walks must effectively "aim" for height -1 or 1 at step n.
    - Forbidden d(n) are values that would cause a dot product of 0
      for any such valid walk.

    Optimizations:
    - Uses Bitsets (integers) with an OFFSET to handle negative partial sums.
    - Exploits symmetry: Forbidden set for target 1 is identical to target -1.
    - Geometric Pruning: Discards paths too far away to reach targets.
    """
    if len(d_initial) < 2:
        raise ValueError("d_initial must contain at least 2 elements")

    if max_n < len(d_initial):
        return d_initial[:max_n]

    # d is 1-based: d[0] is unused, d[1]=d_initial[0], d[2]=d_initial[1], etc.
    d = [0] + list(d_initial)

    # --- DYNAMIC OFFSET ---
    # Conservative estimate for bitmask offset to handle negative numbers
    # 40 billion bits ~ 5GB RAM
    SAFE_RAM_LIMIT = 40_000_000_000
    estimated_max_sum = 1 << (max_n + 2)
    OFFSET = min(estimated_max_sum, SAFE_RAM_LIMIT)

    # Initial state: Height 0, Sum 0.
    layers: Dict[int, int] = {0: 1 << OFFSET}

    current_step = 0

    def advance_layers(current_layers: Dict[int, int], weight: int, step_index: int) -> Dict[int, int]:
        new_layers: Dict[int, int] = {}

        # Pruning: Max distance from {-1, 1} allowed
        steps_remaining = max_n - (step_index + 1)
        max_dist = steps_remaining + 2

        active_heights = sorted(list(current_layers.keys()))

        for h in active_heights:
            mask = current_layers.pop(h)

            for delta in (-1, 0, 1):
                nh = h + delta

                # Pruning check
                if abs(nh) > max_dist:
                    continue

                shift = nh * weight

                if shift > 0:
                    shifted_mask = mask << shift
                elif shift < 0:
                    shifted_mask = mask >> (-shift)
                else:
                    shifted_mask = mask

                if nh in new_layers:
                    new_layers[nh] |= shifted_mask
                else:
                    new_layers[nh] = shifted_mask

            del mask

        return new_layers

    # --- INITIALIZATION ---
    # We need the state at step n-1 to compute d(n).
    # The loop will start at n = len(d_initial) + 1
    # For that iteration, we need the state at step n-1 = len(d_initial)
    # So we advance through d[1], d[2], ..., d[len(d_initial)]
    for i in range(1, len(d_initial) + 1):
        layers = advance_layers(layers, d[i], current_step)
        current_step += 1

    # --- MAIN LOOP ---
    start_n = len(d_initial) + 1
    for n in range(start_n, max_n + 1):
        # 1. IDENTIFY FORBIDDEN SUMS
        # We are at step n-1.
        # Valid walks end at step n with height -1 or 1.
        # To reach -1 at step n, we must be at {-2, -1, 0} at step n-1.
        # To reach 1 at step n, we must be at {0, 1, 2} at step n-1.
        # Due to symmetry (F_1 = F_-1), we just need the union of reachable sums
        # from {-2, -1, 0}.

        forbidden_mask = 0
        for h in [-2, -1, 0]:
            if h in layers:
                forbidden_mask |= layers[h]

        # 2. FIND SMALLEST MISSING POSITIVE INTEGER
        # Slice mask starting at OFFSET + 1 to check positive integers
        positive_search_space = forbidden_mask >> (OFFSET + 1)

        # Hardware gap finding
        lowest_zero_bit = (~positive_search_space) & (positive_search_space + 1)
        k = lowest_zero_bit.bit_length()

        if k == 0:
            k = 1

        d.append(k)

        # 3. ADVANCE STATE
        # Move from step n-1 to step n using the new d[n]
        if n < max_n:
            weight_to_add = d[n]  # Current weight
            layers = advance_layers(layers, weight_to_add, current_step)
            current_step += 1

    return d[1:]


if __name__ == "__main__":
    # Benchmark
    N = 25
    SEED = [1, 5, 7]
    print(f"Generating Smooth Walk sequence for seed {SEED} up to n={N}...")

    start = time.perf_counter()
    res = optimize_smooth_walk_greedy(N, SEED)
    dt = time.perf_counter() - start

    print(f"Time: {dt:.4f}s")
    print(f"Result (last 10): {res}")
    print(f"Full Result: {res}")
