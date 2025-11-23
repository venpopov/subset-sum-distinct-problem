# Benchmark Report: motzkin_greedy vs motzkin_greedy_v2 vs motzkin_greedy_v3

**Date:** 2025-11-23  
**Seed:** (1, 1) - Fixed for reproducibility  
**Machine:** macOS  

## Measured Results (psutil / OS-level RSS)

The following table shows the exact run output measured with `psutil` (OS-level resident memory, RSS).

================================================================================
BENCHMARK: motzkin_greedy vs motzkin_greedy_v2 vs motzkin_greedy_v3
================================================================================
Seed: (1, 1)
Time measurements: average of N runs (N depends on max_n)
Memory measurements: Peak OS-level memory (RSS) from run(s)

max_n  Version              Time (s)     Memory (MB)  Status    
--------------------------------------------------------------------------------
5      motzkin_greedy       0.000025     22.09        OK        
5      motzkin_greedy_v2    0.000026     22.09        OK        
5      motzkin_greedy_v3    0.000028     22.09        OK        

10     motzkin_greedy       0.001756     22.25        OK        
10     motzkin_greedy_v2    0.002345     22.28        OK        
10     motzkin_greedy_v3    0.000049     22.28        OK        

15     motzkin_greedy       0.269153     48.12        OK        
15     motzkin_greedy_v2    0.179490     48.39        OK        
15     motzkin_greedy_v3    0.001106     48.42        OK        

20     motzkin_greedy       51.458132    5307.52      OK        
20     motzkin_greedy_v2    29.956311    5319.72      OK        
20     motzkin_greedy_v3    0.301316     5313.69      OK        

================================================================================
SUMMARY
================================================================================

max_n = 5
  Results correctness: ✓ ALL MATCH
  Time ratios (relative to v1):
    v1: 0.000025s (baseline)
    v2: 0.000026s (1.02x)
    v3: 0.000028s (1.11x)
  Memory ratios (relative to v1):
    v1: 22.09MB (baseline)
    v2: 22.09MB (1.00x)
    v3: 22.09MB (1.00x)

max_n = 10
  Results correctness: ✓ ALL MATCH
  Time ratios (relative to v1):
    v1: 0.001756s (baseline)
    v2: 0.002345s (1.34x)
    v3: 0.000049s (0.03x)
  Memory ratios (relative to v1):
    v1: 22.25MB (baseline)
    v2: 22.28MB (1.00x)
    v3: 22.28MB (1.00x)

max_n = 15
  Results correctness: ✓ ALL MATCH
  Time ratios (relative to v1):
    v1: 0.269153s (baseline)
    v2: 0.179490s (0.67x)
    v3: 0.001106s (0.00x)
  Memory ratios (relative to v1):
    v1: 48.12MB (baseline)
    v2: 48.39MB (1.01x)
    v3: 48.42MB (1.01x)

max_n = 20
  Results correctness: ✓ ALL MATCH
  Time ratios (relative to v1):
    v1: 51.458132s (baseline)
    v2: 29.956311s (0.58x)
    v3: 0.301316s (0.01x)
  Memory ratios (relative to v1):
    v1: 5307.52MB (baseline)
    v2: 5319.72MB (1.00x)
    v3: 5313.69MB (1.00x)

## Key observations

- All three implementations produce identical correct sequences for the tested values.
- **v3 (`motzkin_greedy_optimized`) is dramatically faster:** at `max_n=20`, it's **170x faster than v1** and **99x faster than v2**.
- Time performance at max_n=20:
  - v1: 51.46 seconds
  - v2: 29.96 seconds  
  - v3: 0.30 seconds (sub-second!)
- All three have similar OS-level memory (RSS) at `max_n=20`: ~5.3 GB.
- The v3 bitset DP approach achieves massive speedup while maintaining the same memory profile.

## Analysis of the large RSS at max_n=20

- The psutil RSS values indicate the whole process resident memory reported by the OS — this includes:
  - Python interpreter memory and allocator overhead
  - C-level allocations (extensions, internal buffers)
  - Fragmented memory that the Python allocator may not return to the OS immediately

- Possible causes for the ~5.3 GB RSS during `max_n=20`:
  - Massive growth of temporary/intermediate data while enumerating Motzkin profiles (combinatorial explosion around n=20).
  - Deep recursion or many simultaneous frames/closures that hold references to large objects.
  - Python memory allocator fragmentation: even if objects are freed, the allocator may hold pages.

## Recommendations

**For production use:**
- **Use v3 (`motzkin_greedy_optimized`)** - it's the clear winner with 100-170x speedup over v1/v2 at large n.
- v3 uses bitset DP which avoids repeatedly enumerating Motzkin paths, achieving dramatic performance gains.
- Memory usage is comparable across all versions (~5.3 GB at max_n=20), so the speed improvement comes with no memory cost.

**Why v3 is so much faster:**
- Uses Python integers as bitsets to represent sets of reachable sums.
- Layer-by-layer DP that merges paths at the same height.
- Incremental state that persists across n (no recomputation).
- Avoids the combinatorial explosion of repeatedly generating Motzkin profiles.

**When to use v1 or v2:**
- For understanding/verification of the algorithm (clearer code structure).
- For small values of max_n where performance doesn't matter.
- When you need explicit profile enumeration for analysis.

## Next steps for memory optimization

If the ~5.3 GB RSS at max_n=20 is still a concern:

1. Profile to find the exact allocation hotspot:
```bash
pip install memory-profiler
mprof run python src/benchmark_motzkin.py
mprof plot
```

2. Try incremental probing to find the RSS jump threshold (e.g., test max_n from 16 to 20).

3. Potential optimizations:
   - Use `gmpy2` or similar for more efficient large integer bitset operations.
   - Implement garbage collection hints between n iterations.
   - Profile and optimize the bitset shift/or operations in v3.

## How to reproduce the exact numbers

1. Ensure `psutil` is installed: `pip install psutil`
2. Run the benchmark script: `python src/benchmark_motzkin.py`

The script now reports OS-level RSS (psutil) as displayed above.

