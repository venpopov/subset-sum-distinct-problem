"""
Benchmark script comparing motzkin_greedy and motzkin_greedy_v2.

Measures:
- Execution time (using time.perf_counter)
- Peak memory usage (using psutil for OS-level measurements)
- Results correctness

Parametrized over max_n values: 5, 10, 15, 20
Fixed seed: (1, 1) for reproducibility

NOTE: Memory measurements reflect actual OS memory usage (RSS), not just
Python's internal tracemalloc, since garbage collection may not immediately
free memory. This gives a more realistic picture of resource consumption.
"""

import time
import tracemalloc
import gc
from typing import List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from motzkin_greedy import motzkin_greedy, motzkin_greedy_v2
from motzkin_greedy_v3 import motzkin_greedy_optimized

# Try to use psutil for accurate OS-level memory measurements
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, using tracemalloc instead (less accurate)")


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
    # Force garbage collection before starting
    gc.collect()
    time.sleep(0.1)  # Give OS time to settle

    if PSUTIL_AVAILABLE:
        process = psutil.Process()

    # Measure time and memory on first run
    if PSUTIL_AVAILABLE:
        process.memory_info()  # Force update

    times = []
    peak_memory_mb = 0
    result = None

    for i in range(num_runs):
        if PSUTIL_AVAILABLE:
            gc.collect()

        start = time.perf_counter()
        result = func(max_n, d1, d2)
        end = time.perf_counter()
        times.append(end - start)

        if PSUTIL_AVAILABLE:
            current_mem = process.memory_info().rss / (1024 * 1024)
            peak_memory_mb = max(peak_memory_mb, current_mem)

    if not PSUTIL_AVAILABLE:
        # Fallback to tracemalloc
        tracemalloc.start()
        result = func(max_n, d1, d2)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak / (1024 * 1024)

    avg_time = sum(times) / len(times)

    return avg_time, peak_memory_mb, result


def main() -> None:
    max_n_values = [5, 10, 15, 20]
    d1, d2 = 1, 1  # Fixed seed for reproducibility

    print("=" * 80)
    print("BENCHMARK: motzkin_greedy vs motzkin_greedy_v2 vs motzkin_greedy_v3")
    print("=" * 80)
    print(f"Seed: ({d1}, {d2})")
    print("Time measurements: average of N runs (N depends on max_n)")
    if PSUTIL_AVAILABLE:
        print("Memory measurements: Peak OS-level memory (RSS) from run(s)")
    else:
        print("Memory measurements: Peak memory from tracemalloc (less accurate than OS-level)")
        print("  Install psutil for more accurate measurements: pip install psutil")
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

        # Benchmark v3
        try:
            time_v3, mem_v3, result_v3 = benchmark_function(motzkin_greedy_optimized, max_n, d1, d2, num_runs=num_runs)
            results_by_n[max_n]["v3"] = {
                "time": time_v3,
                "memory": mem_v3,
                "result": result_v3,
                "status": "OK",
            }
            print(f"{max_n:<6} {'motzkin_greedy_v3':<20} {time_v3:<12.6f} {mem_v3:<12.2f} {'OK':<10}")
        except Exception as e:
            results_by_n[max_n]["v3"] = {
                "status": f"ERROR: {e}",
            }
            print(f"{max_n:<6} {'motzkin_greedy_v3':<20} {'ERROR':<12} {'N/A':<12} {str(e)[:10]:<10}")

        print()

    # Summary and comparison
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for max_n in max_n_values:
        v1_data = results_by_n[max_n].get("v1", {})
        v2_data = results_by_n[max_n].get("v2", {})
        v3_data = results_by_n[max_n].get("v3", {})

        all_ok = v1_data.get("status") == "OK" and v2_data.get("status") == "OK" and v3_data.get("status") == "OK"

        if all_ok:
            # Check correctness
            if v1_data["result"] == v2_data["result"] == v3_data["result"]:
                correctness = "✓ ALL MATCH"
            else:
                correctness = "✗ MISMATCH"

            # Compute ratios
            time_ratio_v2 = v2_data["time"] / v1_data["time"]
            time_ratio_v3 = v3_data["time"] / v1_data["time"]
            mem_ratio_v2 = v2_data["memory"] / v1_data["memory"]
            mem_ratio_v3 = v3_data["memory"] / v1_data["memory"]

            print(f"\nmax_n = {max_n}")
            print(f"  Results correctness: {correctness}")
            print(f"  Time ratios (relative to v1):")
            print(f"    v1: {v1_data['time']:.6f}s (baseline)")
            print(f"    v2: {v2_data['time']:.6f}s ({time_ratio_v2:.2f}x)")
            print(f"    v3: {v3_data['time']:.6f}s ({time_ratio_v3:.2f}x)")
            print(f"  Memory ratios (relative to v1):")
            print(f"    v1: {v1_data['memory']:.2f}MB (baseline)")
            print(f"    v2: {v2_data['memory']:.2f}MB ({mem_ratio_v2:.2f}x)")
            print(f"    v3: {v3_data['memory']:.2f}MB ({mem_ratio_v3:.2f}x)")
        else:
            print(f"\nmax_n = {max_n}: ERRORS OCCURRED")
            if v1_data.get("status") != "OK":
                print(f"  v1: {v1_data.get('status')}")
            if v2_data.get("status") != "OK":
                print(f"  v2: {v2_data.get('status')}")
            if v3_data.get("status") != "OK":
                print(f"  v3: {v3_data.get('status')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
