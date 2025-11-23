"""
Generate Motzkin–greedy difference sequences d(n) and corresponding P_n
for arbitrary seeds, optionally checking the DSS property.

This script processes multiple seeds in parallel, with each seed assigned to a
separate worker process. All available CPU cores are utilized by default.

COMMAND-LINE ARGUMENTS:

    --seeds SEED [SEED ...]
        Space-separated list of seeds in format 'd1,d2'.
        Example: --seeds 2,1 1,1 3,2
        Default: All 13 standard seeds (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (2,3)
                                          (3,1) (3,2) (3,3) (4,1) (4,3) (4,4)

    --max-n N
        Maximum index n for the difference sequence to generate (default: 20).
        For each seed, generates d(1), d(2), ..., d(n) and computes P_n.

    --out-dir PATH
        Output directory for all generated files (default: 'data').
        Creates the directory if it doesn't exist.
        Output files:
            - seed_{d1}_{d2}_d.csv       : Difference sequence as comma-separated values
            - P_sets_seed_{d1}_{d2}.json : Per-n data (P values, SSD flags, collision info)
            - summary_seed_{d1}_{d2}.csv : Tabular summary (n, d_n, maxP, SSD_flag)

    --check-ssd
        Enable brute-force DSS (distinct subset sum) verification.
        Without this flag, SSD_flag will be -1 in output files.
        When enabled, uses two independent checkers:
            - Combinatorial enumeration (itertools.combinations)
            - Bitmask enumeration
        Note: Significantly increases computation time for large n.

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
from utils import P_from_d_prefix, parse_seed
from ssd_check import (
    is_distinct_subset_sum_combinations,
    is_distinct_subset_sum_bitmask,
)

# Setup logging for multiprocessing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def generate_for_seed(seed: str, max_n: int, out_dir: Path, check_ssd: bool) -> Tuple[str, bool]:
    """
    Generate sequences for a single seed. Returns (seed_str, success).
    """
    try:
        d1, d2 = parse_seed(seed)
        logging.info(f"Starting: Seed (d1,d2) = ({d1},{d2}), max_n = {max_n}")

        d_seq = motzkin_greedy_optimized(max_n, d1, d2)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Save d-sequence
        d_file = out_dir / f"seed_{d1}_{d2}_d.csv"
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

            if check_ssd:
                ok_c, info_c = is_distinct_subset_sum_combinations(Pn)
                ok_b, info_b = is_distinct_subset_sum_bitmask(Pn)
                # they should agree
                if ok_c != ok_b:
                    logging.warning(f"Seed ({d1},{d2}), n={n}: SSD checkers disagree")
                ok = ok_c and ok_b
                ssd_flags.append(1 if ok else 0)
                record["ssd"] = ok
                if not ok:
                    record["collision_combinations"] = info_c
                    record["collision_bitmask"] = info_b

            P_data[n] = record

        # Save P data as JSON
        P_file = out_dir / f"P_sets_seed_{d1}_{d2}.json"
        with P_file.open("w") as f:
            json.dump(P_data, f, indent=2)

        # Save summary table
        summary_file = out_dir / f"summary_seed_{d1}_{d2}.csv"
        with summary_file.open("w") as f:
            f.write("n,d_n,maxP,SSD_flag\n")
            for n in range(1, max_n + 1):
                d_n = d_seq[n - 1]
                maxP = max_P_list[n - 1]
                ssd_flag = ssd_flags[n - 1] if check_ssd else -1
                f.write(f"{n},{d_n},{maxP},{ssd_flag}\n")

        logging.info(f"Completed: Seed ({d1},{d2})")
        return (seed, True)
    except Exception as e:
        logging.error(f"Failed to process seed {seed}: {e}")
        return (seed, False)


def _worker_wrapper(args: Tuple[str, int, Path, bool]) -> Tuple[str, bool]:
    """Wrapper for multiprocessing pool."""
    seed, max_n, out_dir, check_ssd = args
    return generate_for_seed(seed, max_n, out_dir, check_ssd)


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
        help="If set, run brute-force DSS checks for each P_n.",
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

    logging.info(f"Starting parallel generation for {len(args.seeds)} seed(s)")
    logging.info(f"Using {num_workers or 'auto'} worker(s)")

    # Prepare work items
    work_items = [(seed, args.max_n, out_dir, args.check_ssd) for seed in args.seeds]

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
