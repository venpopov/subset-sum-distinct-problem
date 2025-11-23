"""
Benchmark script comparing motzkin_greedy and motzkin_greedy_v2.

Measures:
- Execution time
- Peak memory usage
- Results correctness

Parametrized over max_n values: 5, 10, 15, 20
Fixed seed: (1, 1) for reproducibility
"""

import time
import tracemalloc
from typing import List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from motzkin_greedy import motzkin_greedy, motzkin_greedy_v2


def benchmark_function(
    func,
    max_n: int,
    d1: int,
    d2: int,
    num_runs: int = 1,
) -> Tuple[float, float, List[int]]:
    """
    Benchmark a function measuring time and peak memory.

    Args:
        func: Function to benchmark
        max_n: Maximum n parameter
        d1, d2: Seed parameters
        num_runs: Number of times to run (for averaging time, memory is from single run)

    Returns:
        (avg_time, peak_memory_mb, result)
    """
    # Warmup (helps with JIT, if any)
    func(max_n, d1, d2)

    # Measure memory on a single run
    tracemalloc.start()
    result = func(max_n, d1, d2)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak / (1024 * 1024)

    # Measure time over multiple runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(max_n, d1, d2)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    return avg_time, peak_memory_mb, result


def main() -> None:
    max_n_values = [5, 10, 15, 20]
    d1, d2 = 1, 1  # Fixed seed for reproducibility

    print("=" * 80)
    print("BENCHMARK: motzkin_greedy vs motzkin_greedy_v2")
    print("=" * 80)
    print(f"Seed: ({d1}, {d2})")
    print("Time measurements: average of N runs (N depends on max_n)")
    print("Memory measurements: peak from single run")
    print()

    # Header
    print(f"{'max_n':<6} {'Version':<20} {'Time (s)':<12} {'Memory (MB)':<12} {'Status':<10}")
    print("-" * 80)

    results_by_n = {}

    for max_n in max_n_values:
        # Use fewer runs for larger max_n values (they take longer)
        num_runs = 1 if max_n >= 15 else 3
        results_by_n[max_n] = {}

        # Benchmark v1
        try:
            time_v1, mem_v1, result_v1 = benchmark_function(motzkin_greedy, max_n, d1, d2, num_runs=num_runs)
            results_by_n[max_n]["v1"] = {
                "time": time_v1,
                "memory": mem_v1,
                "result": result_v1,
                "status": "OK",
            }
            print(f"{max_n:<6} {'motzkin_greedy':<20} {time_v1:<12.6f} {mem_v1:<12.2f} {'OK':<10}")
        except Exception as e:
            results_by_n[max_n]["v1"] = {
                "status": f"ERROR: {e}",
            }
            print(f"{max_n:<6} {'motzkin_greedy':<20} {'ERROR':<12} {'N/A':<12} {str(e)[:10]:<10}")

        # Benchmark v2
        try:
            time_v2, mem_v2, result_v2 = benchmark_function(motzkin_greedy_v2, max_n, d1, d2, num_runs=num_runs)
            results_by_n[max_n]["v2"] = {
                "time": time_v2,
                "memory": mem_v2,
                "result": result_v2,
                "status": "OK",
            }
            print(f"{max_n:<6} {'motzkin_greedy_v2':<20} {time_v2:<12.6f} {mem_v2:<12.2f} {'OK':<10}")
        except Exception as e:
            results_by_n[max_n]["v2"] = {
                "status": f"ERROR: {e}",
            }
            print(f"{max_n:<6} {'motzkin_greedy_v2':<20} {'ERROR':<12} {'N/A':<12} {str(e)[:10]:<10}")

        print()

    # Summary and comparison
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for max_n in max_n_values:
        v1_data = results_by_n[max_n].get("v1", {})
        v2_data = results_by_n[max_n].get("v2", {})

        if v1_data.get("status") == "OK" and v2_data.get("status") == "OK":
            # Check correctness
            if v1_data["result"] == v2_data["result"]:
                correctness = "✓ MATCH"
            else:
                correctness = "✗ MISMATCH"

            # Compute ratios
            time_ratio = v2_data["time"] / v1_data["time"]
            mem_ratio = v2_data["memory"] / v1_data["memory"]

            print(f"\nmax_n = {max_n}")
            print(f"  Results correctness: {correctness}")
            print(f"  Time ratio (v2/v1):   {time_ratio:.2f}x")
            print(f"    v1: {v1_data['time']:.6f}s, v2: {v2_data['time']:.6f}s")
            print(f"  Memory ratio (v2/v1): {mem_ratio:.2f}x")
            print(f"    v1: {v1_data['memory']:.2f}MB, v2: {v2_data['memory']:.2f}MB")
        else:
            print(f"\nmax_n = {max_n}: ERRORS OCCURRED")
            if v1_data.get("status") != "OK":
                print(f"  v1: {v1_data.get('status')}")
            if v2_data.get("status") != "OK":
                print(f"  v2: {v2_data.get('status')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
