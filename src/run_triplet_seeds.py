#!/usr/bin/env python3
"""
Run generate_sequences.py over all triplet seeds from (1,1,1) to (6,6,6).

This script generates all combinations of seeds (a,b,c) where 1 ≤ a,b,c ≤ 6
and runs generate_sequences.py for each seed using multiprocessing.

Usage:
    python run_triplet_seeds.py [--max-n MAX_N] [--workers WORKERS] [--out-dir OUT_DIR] [--no-check-ssd]

Arguments:
    --max-n MAX_N       Maximum n value for sequence generation (default: 30)
    --workers WORKERS   Number of parallel workers (default: 12)
    --out-dir OUT_DIR   Output directory for results (default: data/triplet_seeds)
    --no-check-ssd      Skip SSD checking (faster but no verification)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from itertools import product
import multiprocessing as mp
from typing import Tuple


def run_seed(args: Tuple[str, int, Path, bool]) -> Tuple[str, bool]:
    """
    Run generate_sequences.py for a single seed.
    
    Args:
        args: Tuple of (seed_str, max_n, out_dir, check_ssd)
    
    Returns:
        Tuple of (seed_str, success)
    """
    seed_str, max_n, out_dir, check_ssd = args
    
    cmd = [
        sys.executable,
        "src/generate_sequences.py",
        "--workers", "1",
        "--seeds", seed_str,
        "--max-n", str(max_n),
        "--out-dir", str(out_dir)
    ]
    
    if check_ssd:
        cmd.append("--check-ssd")
    
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Completed seed {seed_str}")
        return (seed_str, True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed seed {seed_str}: {e}")
        print(f"  stderr: {e.stderr}")
        return (seed_str, False)


def main():
    parser = argparse.ArgumentParser(
        description="Run generate_sequences.py for all triplet seeds (1,1,1) to (6,6,6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=30,
        help="Maximum n value for sequence generation (default: 30)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel workers (default: 12)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/triplet_seeds"),
        help="Output directory for results (default: data/triplet_seeds)"
    )
    parser.add_argument(
        "--no-check-ssd",
        action="store_true",
        help="Skip SSD checking (faster but no verification)"
    )
    
    args = parser.parse_args()
    
    # Generate all triplet seeds from (1,1,1) to (6,6,6)
    seeds = []
    for a, b, c in product(range(1, 7), repeat=3):
        seeds.append(f"{a},{b},{c}")
    
    total_seeds = len(seeds)
    print(f"Generated {total_seeds} triplet seeds from (1,1,1) to (6,6,6)")
    print(f"Running with {args.workers} workers, max_n={args.max_n}")
    print(f"Output directory: {args.out_dir}")
    print(f"SSD checking: {'disabled' if args.no_check_ssd else 'enabled'}")
    print()
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for all seeds
    check_ssd = not args.no_check_ssd
    task_args = [(seed, args.max_n, args.out_dir, check_ssd) for seed in seeds]
    
    # Run in parallel
    print("Starting parallel execution...")
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(run_seed, task_args)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for _, success in results if success)
    failed = total_seeds - successful
    
    print(f"Total seeds: {total_seeds}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed seeds:")
        for seed, success in results:
            if not success:
                print(f"  - {seed}")
        sys.exit(1)
    else:
        print("\n✓ All seeds completed successfully!")


if __name__ == "__main__":
    main()
