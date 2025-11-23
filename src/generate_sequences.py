"""
Generate Motzkin–greedy difference sequences d(n) and corresponding P_n
for arbitrary seeds, optionally checking the DSS property.

This script processes multiple seeds in parallel, with each seed assigned to a
separate worker process. All available CPU cores are utilized by default.

COMMAND-LINE ARGUMENTS:

    --seeds SEED [SEED ...]
        Space-separated list of seeds in format 'd1,d2,...'.
        Each seed must have at least 2 comma-separated integers.
        Example: --seeds 2,1 1,1 3,2 or --seeds 1,2,3 2,1,1
        Default: All 13 standard seeds (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (2,3)
                                          (3,1) (3,2) (3,3) (4,1) (4,3) (4,4)

    --max-n N
        Maximum index n for the difference sequence to generate (default: 20).
        For each seed, generates d(1), d(2), ..., d(n) and computes P_n.

    --out-dir PATH
        Output directory for all generated files (default: 'data').
        Creates the directory if it doesn't exist.
        Output files (seed elements joined with underscores):
            - seed_{d1}_{d2}...{dn}_d.csv       : Difference sequence as comma-separated values
            - P_sets_seed_{d1}_{d2}...{dn}.json : Per-n data (P values, SSD flags, collision info)
            - summary_seed_{d1}_{d2}...{dn}.csv : Tabular summary (n, d_n, maxP, SSD_flag)

    --check-ssd
        Enable DSS (distinct subset sum) verification using the optimized bitset algorithm.
        Without this flag, SSD_flag will be -1 in output files.
        This is the fast method: O(n * sum(P)) complexity.

    --check-ssd-slow
        Enable brute-force DSS verification using two independent slow checkers:
            - Combinatorial enumeration (itertools.combinations)
            - Bitmask enumeration
        Both have O(2^n) complexity.
        Note: Significantly slower than --check-ssd, use only for verification.

    --workers N
        Number of parallel worker processes (default: auto-detect CPU count).
        Each seed is processed by a separate worker.
        Example: --workers 12 (use 12 cores maximum)

USAGE EXAMPLES:

    1. Generate all 13 standard seeds (uses all available cores):
       python src/generate_sequences.py --check-ssd

    2. Generate specific seeds with 12 workers:
       python src/generate_sequences.py --seeds 2,1 1,1 --workers 12 --check-ssd

    3. Generate without SSD checking (faster):
       python src/generate_sequences.py --max-n 30

    4. Custom output directory:
       python src/generate_sequences.py --out-dir results/experiment1 --check-ssd

    5. Single seed (still uses multiprocessing):
       python src/generate_sequences.py --seeds 2,1 --check-ssd

OUTPUT SUMMARY:

For each processed seed, the script generates three files in <out-dir>:

    1. seed_{d1}_{d2}_d.csv
       - Single row containing the difference sequence
       - Comma-separated values: d(1),d(2),...,d(n)

    2. P_sets_seed_{d1}_{d2}.json
       - JSON object with n as keys (1 to max_n)
       - Each entry contains:
           n: index
           P: decreasing set of partial sums
           maxP: maximum element in P_n
           ssd: boolean (true if DSS, only if --check-ssd)
           collision_combinations: collision info (if not DSS)
           collision_bitmask: collision info (if not DSS)

    3. summary_seed_{d1}_{d2}.csv
       - CSV table with columns: n, d_n, maxP, SSD_flag
       - One row per index, useful for analysis and plotting

PARALLEL EXECUTION:

This script uses Python's multiprocessing.Pool to distribute seed processing
across available CPU cores. Each seed runs independently, so:
- One seed's failure does not halt processing of others
- Processing is load-balanced across workers
- Perfect for batch processing many seeds

Logging includes:
- Start/completion timestamps for each seed
- Warnings for SSD checker disagreements
- Final summary with success/failure status for each seed
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool
import logging

from motzkin_greedy import motzkin_greedy_optimized
from utils import P_from_d_prefix, parse_seed_list
from ssd_check import (
    is_distinct_subset_sum_combinations,
    is_distinct_subset_sum_bitmask,
    is_distinct_subset_sum_optimized,
)

# Setup logging for multiprocessing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def generate_for_seed(
    seed: str, max_n: int, out_dir: Path, check_ssd: bool, check_ssd_slow: bool = False
) -> Tuple[str, bool]:
    """
    Generate sequences for a single seed. Returns (seed_str, success).
    """
    try:
        d_initial = parse_seed_list(seed)
        seed_label = "_".join(str(x) for x in d_initial)
        logging.info(f"Starting: Seed {d_initial}, max_n = {max_n}")

        d_seq = motzkin_greedy_optimized(max_n, d_initial)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Save d-sequence
        d_file = out_dir / f"seed_{seed_label}_d.csv"
        with d_file.open("w") as f:
            f.write(",".join(str(x) for x in d_seq) + "\n")

        # Build P_n and optionally check SSD
        P_data: Dict[int, Dict[str, Any]] = {}
        ssd_flags: List[int] = []
        max_P_list: List[int] = []

        for n in range(1, max_n + 1):
            d_prefix = d_seq[:n]
            Pn = P_from_d_prefix(d_prefix)
            maxP = Pn[0]
            max_P_list.append(maxP)

            record: Dict[str, Any] = {
                "n": n,
                "P": Pn,
                "maxP": maxP,
            }

            if check_ssd or check_ssd_slow:
                if check_ssd_slow:
                    # Use slow brute-force checkers for verification
                    ok_c, info_c = is_distinct_subset_sum_combinations(Pn)
                    ok_b, info_b = is_distinct_subset_sum_bitmask(Pn)
                    # they should agree
                    if ok_c != ok_b:
                        logging.warning(f"Seed {d_initial}, n={n}: SSD checkers disagree")
                    ok = ok_c and ok_b
                    ssd_flags.append(1 if ok else 0)
                    record["ssd"] = ok
                    if not ok:
                        record["collision_combinations"] = info_c
                        record["collision_bitmask"] = info_b
                else:
                    # Use fast optimized checker
                    ok, info = is_distinct_subset_sum_optimized(Pn)
                    ssd_flags.append(1 if ok else 0)
                    record["ssd"] = ok
                    if not ok:
                        record["collision_optimized"] = info

            P_data[n] = record

        # Save P data as JSON
        P_file = out_dir / f"P_sets_seed_{seed_label}.json"
        with P_file.open("w") as f:
            json.dump(P_data, f, indent=2)

        # Save summary table
        summary_file = out_dir / f"summary_seed_{seed_label}.csv"
        with summary_file.open("w") as f:
            f.write("n,d_n,maxP,SSD_flag\n")
            for n in range(1, max_n + 1):
                d_n = d_seq[n - 1]
                maxP = max_P_list[n - 1]
                ssd_flag = ssd_flags[n - 1] if check_ssd else -1
                f.write(f"{n},{d_n},{maxP},{ssd_flag}\n")

        logging.info(f"Completed: Seed {d_initial}")
        return (seed, True)
    except Exception as e:
        logging.error(f"Failed to process seed {seed}: {e}")
        return (seed, False)


def _worker_wrapper(args: Tuple[str, int, Path, bool, bool]) -> Tuple[str, bool]:
    """Wrapper for multiprocessing pool."""
    seed, max_n, out_dir, check_ssd, check_ssd_slow = args
    return generate_for_seed(seed, max_n, out_dir, check_ssd, check_ssd_slow)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Motzkin–greedy d(n) and P_n for multiple seeds in parallel.")
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["1,1", "1,2", "1,3", "1,4", "2,1", "2,2", "2,3", "3,1", "3,2", "3,3", "4,1", "4,3", "4,4"],
        help="Seeds as space-separated 'd1,d2' pairs. Default: all 13 standard seeds.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=20,
        help="Maximum n to generate (default: 20).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory (default: 'data').",
    )
    parser.add_argument(
        "--check-ssd",
        action="store_true",
        help="If set, run fast optimized DSS checks for each P_n.",
    )
    parser.add_argument(
        "--check-ssd-slow",
        action="store_true",
        help="If set, run slow brute-force DSS checks (combinations + bitmask) for verification.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    num_workers = args.workers

    # Validate that both check-ssd flags aren't set together
    if args.check_ssd and args.check_ssd_slow:
        logging.warning("Both --check-ssd and --check-ssd-slow specified. Using slow checkers.")
        args.check_ssd = False

    logging.info(f"Starting parallel generation for {len(args.seeds)} seed(s)")
    logging.info(f"Using {num_workers or 'auto'} worker(s)")
    if args.check_ssd:
        logging.info("SSD checking: ENABLED (optimized bitset method)")
    elif args.check_ssd_slow:
        logging.info("SSD checking: ENABLED (slow brute-force methods)")
    else:
        logging.info("SSD checking: DISABLED")

    # Prepare work items
    work_items = [(seed, args.max_n, out_dir, args.check_ssd, args.check_ssd_slow) for seed in args.seeds]

    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(_worker_wrapper, work_items)

    # Print summary
    logging.info("\n" + "=" * 50)
    logging.info("SUMMARY")
    logging.info("=" * 50)
    successful = sum(1 for _, success in results if success)
    logging.info(f"Successfully completed: {successful}/{len(args.seeds)}")

    for seed, success in results:
        status = "✓ OK" if success else "✗ FAILED"
        logging.info(f"  {status}: Seed {seed}")

    if successful == len(args.seeds):
        logging.info("All seeds processed successfully!")
    else:
        logging.warning(f"{len(args.seeds) - successful} seed(s) failed.")


if __name__ == "__main__":
    main()
