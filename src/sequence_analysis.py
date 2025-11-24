import pandas as pd
from pathlib import Path
from typing import List, Callable, Tuple
import logging
from multiprocessing import Pool
import pickle

from motzkin_greedy import optimize_motzkin_greedy
from smooth_walk_greedy import optimize_smooth_walk_greedy
from utils import P_from_d_prefix
from ssd_check import is_distinct_subset_sum_optimized

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Cache file location
_CACHE_FILE = Path("data/.ssd_cache.pkl")


# Load cache from disk if it exists
def _load_cache():
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                logging.info(f"Loaded cache with {len(cache)} entries from {_CACHE_FILE}")
                return cache
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}. Starting with empty cache.")
            return {}
    return {}


def _save_cache(cache):
    """Save cache to disk."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        logging.debug(f"Saved cache with {len(cache)} entries to {_CACHE_FILE}")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")


# Global cache for SSD results
_SSD_CACHE = _load_cache()


def generate_ssd_df(
    d_seed: List[int], max_n: int, optimizer: Callable = optimize_motzkin_greedy, check_ssd: bool = True
) -> pd.DataFrame:
    """
    Generates a greedy sequence based on a seed and checks for
    Distinct Subset Sum (SSD) properties up to max_n.
    Caches results to avoid redundant computations.

    Args:
        d_seed: Initial seed sequence
        max_n: Maximum n value to compute
        optimizer: Function to use for optimization (default: optimize_motzkin_greedy)
                   Can also pass optimize_smooth_walk_greedy
        check_ssd: Whether to perform SSD check (default: True). If False, SSD_flag will be -1
    """
    # Create a hashable key from arguments
    cache_key = (tuple(d_seed), max_n, optimizer.__name__, check_ssd)

    if cache_key in _SSD_CACHE:
        print("Returning cached result.")
        return _SSD_CACHE[cache_key].copy()

    d_seq = optimizer(max_n, d_seed)

    ssd_flags: List[int] = []
    max_P_list: List[int] = []

    for n in range(1, max_n + 1):
        d_prefix = d_seq[:n]
        Pn = P_from_d_prefix(d_prefix)
        maxP = Pn[0]
        max_P_list.append(maxP)

        if check_ssd:
            ok, info = is_distinct_subset_sum_optimized(Pn)
            ssd_flags.append(1 if ok else 0)
        else:
            ssd_flags.append(-1)

    # Create DataFrame using the collected data
    df = pd.DataFrame({"n": range(1, max_n + 1), "d_n": d_seq, "maxP": max_P_list, "SSD_flag": ssd_flags})

    # Store in cache and save to disk
    _SSD_CACHE[cache_key] = df
    _save_cache(_SSD_CACHE)

    return df


def compare_and_save(
    d_seed: List[int],
    max_n: int,
    optimizer: Callable = optimize_motzkin_greedy,
    d_seed_reference: List[int] = [1, 1],
    return_for_multiprocessing: bool = False,
    check_ssd: bool = True,
) -> pd.DataFrame:
    """
    Generate sequences, compare with reference, and save if condition is met.

    Condition: At least one row (after the first two) has delta <= 0 and SSD == 1.
    If true, saves to CSV with seed in filename.

    Args:
        d_seed: Seed sequence to test
        max_n: Maximum n value
        optimizer: Optimizer function to use
        d_seed_reference: Reference seed (default [1,1])
        return_for_multiprocessing: If True, returns (d_seed, condition_met) tuple
                                    instead of DataFrame for parallel processing
        check_ssd: Whether to perform SSD check (default: True). If False, SSD_flag will be -1

    Returns:
        DataFrame with renamed columns: SSD, d, P, delta (or tuple if return_for_multiprocessing=True)
    """
    try:
        if return_for_multiprocessing:
            logging.info(f"Processing seed {d_seed}")

        df_reference = generate_ssd_df(d_seed_reference, max_n, check_ssd=check_ssd)
        df = generate_ssd_df(d_seed, max_n, optimizer=optimizer, check_ssd=check_ssd)
        df = df.join(df_reference.set_index("n"), on="n", lsuffix="_new", rsuffix="_ref")
        df["maxp_delta"] = df["maxP_new"] - df["maxP_ref"]

        # Create display dataframe
        display_df = df[["SSD_flag_new", "d_n_new", "maxP_new", "maxp_delta"]].copy()
        display_df.columns = ["SSD", "d", "P", "delta"]

        # Check condition 1: rows after first two with delta <= 0 and SSD == 1
        condition_met = (display_df.iloc[2:]["delta"] <= 0) & (abs(display_df.iloc[2:]["SSD"]) == 1)

        # Check condition 2: no more than 2/3 of SSD values are 0
        ssd_zero_ratio = (display_df["SSD"] == 0).sum() / len(display_df)
        ssd_condition_met = ssd_zero_ratio <= 1 / 8

        if condition_met.any() and ssd_condition_met:
            # Create output directory if it doesn't exist
            output_dir = Path("data/special_seeds")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename from seed
            seed_str = "_".join(map(str, d_seed))
            filename = output_dir / f"seed_{seed_str}_n{max_n}.csv"
            display_df.to_csv(filename, index=False)

            if return_for_multiprocessing:
                logging.info(f"✓ Condition met! Saved to {filename}")
            else:
                print(f"✓ Condition met! Saved to {filename}")
        else:
            if return_for_multiprocessing:
                logging.debug(f"✗ Condition not met for seed {d_seed}")
            else:
                print(f"✗ Condition not met for seed {d_seed}")

        # Return appropriate type based on context
        if return_for_multiprocessing:
            return (d_seed, condition_met.any())
        else:
            return display_df

    except Exception as e:
        if return_for_multiprocessing:
            logging.error(f"Failed to process seed {d_seed}: {e}")
            return (d_seed, False)
        else:
            print(f"Failed to process seed {d_seed}: {e}")
            raise


def _worker_wrapper(args: Tuple[List[int], int, Callable, List[int], bool]) -> Tuple[List[int], bool]:
    """Wrapper for multiprocessing pool."""
    d_seed, max_n, optimizer, d_seed_reference, check_ssd = args
    return compare_and_save(
        d_seed, max_n, optimizer, d_seed_reference, return_for_multiprocessing=True, check_ssd=check_ssd
    )


def is_valid_seed(seed: List[int]) -> bool:
    """
    Check if a seed is valid according to these rules:
    1. Cannot have more than two of the same element
    2. If an element appears twice, the duplicates must be adjacent
    3. At least one element must be greater than 8

    Args:
        seed: List of integers to validate

    Returns:
        True if valid, False otherwise
    """
    from collections import Counter

    # Count occurrences of each element
    counts = Counter(seed)

    # Check rule 1: no element can appear more than twice
    if any(count > 2 for count in counts.values()):
        return False

    # Check rule 2: duplicates must be adjacent
    for value, count in counts.items():
        if count == 2:
            # Find indices of this value
            indices = [i for i, x in enumerate(seed) if x == value]
            # Check if they're adjacent
            if indices[1] - indices[0] != 1:
                return False

    # Check rule 3: at least one element must be greater than 7
    if not any(x > 7 for x in seed):
        return False

    return True


if __name__ == "__main__":
    # example use: python src/sequence_analysis.py --workers 12
    # uv run python src/sequence_analysis.py --workers 1 --no-ssd-check --max-n 30
    from itertools import product
    import argparse

    parser = argparse.ArgumentParser(description="Analyze seed sequences in parallel to find special cases.")
    parser.add_argument(
        "--max-n",
        type=int,
        default=25,
        help="Maximum n to generate (default: 25).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count).",
    )
    parser.add_argument(
        "--no-ssd-check",
        action="store_true",
        help="Skip SSD check (sets SSD_flag to -1).",
    )
    args = parser.parse_args()

    max_n = args.max_n
    num_workers = args.workers
    check_ssd = not args.no_ssd_check
    d_seed_reference = [1, 1]
    optimizer = optimize_smooth_walk_greedy

    # Generate all seed vectors: lengths 2-5, values 1-5
    all_seeds = [
        [1, 1, 2, 8, 8, 16, 24, 48, 88, 160],
        [1, 1, 5, 3, 3, 12, 23, 41, 79, 158],
        [1, 1, 8, 4, 4, 16, 24, 44, 88, 156],
        [1, 3, 4, 8, 6, 6, 24, 44, 80, 154],
        [1, 4, 4, 8, 6, 6, 24, 44, 80, 154],
        [1, 5, 8, 4, 4, 16, 24, 44, 88, 156],
        [1, 6, 8, 4, 4, 16, 24, 44, 88, 156],
        [1, 7, 4, 4, 6, 10, 24, 44, 78, 156],
        [1, 8, 4, 4, 6, 10, 24, 44, 78, 156],
        [1, 9, 6, 4, 4, 8, 22, 44, 78, 152],
        [2, 2, 4, 3, 3, 12, 22, 40, 77, 154],
        [2, 2, 4, 6, 9, 3, 22, 40, 80, 151],
        [2, 4, 7, 9, 8, 8, 32, 48, 88, 176],
        [2, 7, 9, 8, 8, 4, 28, 48, 88, 176],
        [2, 9, 7, 8, 8, 4, 28, 48, 88, 176],
        [3, 1, 4, 8, 6, 6, 24, 44, 80, 154],
        [3, 2, 8, 4, 4, 16, 24, 44, 88, 156],
        [3, 3, 7, 5, 5, 14, 24, 43, 91, 177],
        [3, 3, 8, 4, 4, 16, 24, 44, 88, 156],
        [3, 4, 2, 2, 8, 12, 22, 44, 78, 156],
        [3, 4, 4, 8, 6, 6, 24, 44, 80, 154],
        [3, 5, 4, 4, 2, 14, 24, 44, 88, 156],
        [3, 5, 4, 4, 6, 10, 24, 44, 78, 156],
        [3, 6, 2, 4, 4, 16, 24, 44, 88, 156],
        [3, 6, 8, 4, 4, 16, 24, 44, 88, 156],
        [3, 7, 6, 4, 4, 8, 22, 44, 78, 152],
        [3, 7, 6, 8, 8, 4, 28, 48, 88, 176],
        [3, 8, 2, 2, 4, 16, 22, 44, 86, 152],
        [3, 8, 4, 4, 2, 14, 24, 44, 88, 156],
        [3, 8, 4, 4, 6, 10, 24, 44, 78, 156],
        [3, 9, 4, 8, 6, 2, 32, 48, 88, 176],
        [4, 1, 3, 8, 6, 6, 24, 44, 80, 154],
        [4, 2, 2, 3, 5, 12, 22, 39, 78, 151],
        [4, 2, 5, 9, 8, 8, 32, 48, 88, 176],
        [4, 3, 1, 8, 6, 6, 24, 44, 80, 154],
        [4, 4, 8, 3, 3, 6, 24, 44, 77, 154],
        [4, 4, 8, 6, 3, 3, 24, 44, 80, 154],
        [4, 4, 8, 6, 5, 1, 24, 44, 80, 154],
        [4, 6, 6, 7, 5, 8, 20, 46, 79, 158],
        [4, 7, 9, 8, 2, 6, 32, 48, 88, 176],
        [5, 1, 8, 4, 4, 16, 24, 44, 88, 156],
        [5, 3, 2, 2, 4, 11, 22, 39, 76, 148],
        [5, 3, 4, 4, 2, 14, 24, 44, 88, 156],
        [5, 3, 4, 4, 6, 10, 24, 44, 78, 156],
        [5, 4, 2, 2, 8, 12, 22, 44, 78, 156],
        [5, 5, 2, 7, 1, 8, 23, 39, 78, 149],
        [5, 5, 7, 3, 3, 6, 24, 43, 76, 152],
        [5, 6, 2, 4, 4, 16, 24, 44, 88, 156],
        [5, 6, 8, 4, 4, 16, 24, 44, 88, 156],
        [5, 7, 3, 3, 6, 14, 23, 43, 89, 155],
        [5, 7, 4, 8, 2, 6, 32, 48, 88, 176],
        [5, 8, 2, 2, 4, 16, 22, 44, 86, 152],
        [5, 8, 4, 4, 2, 14, 24, 44, 88, 156],
        [5, 9, 2, 8, 8, 4, 28, 48, 88, 176],
        [6, 1, 7, 4, 4, 16, 24, 44, 88, 156],
        [6, 3, 3, 7, 5, 5, 23, 43, 76, 147],
        [6, 3, 3, 8, 4, 4, 25, 44, 77, 150],
        [6, 3, 5, 4, 4, 16, 24, 44, 88, 156],
        [6, 3, 7, 8, 8, 4, 28, 48, 88, 176],
        [6, 3, 9, 4, 8, 8, 32, 48, 88, 176],
        [6, 4, 2, 2, 3, 11, 22, 40, 78, 151],
        [6, 4, 2, 2, 5, 9, 22, 40, 78, 149],
        [6, 4, 2, 2, 9, 5, 22, 40, 78, 151],
        [6, 5, 3, 4, 4, 16, 24, 44, 88, 156],
        [6, 7, 3, 8, 8, 4, 28, 48, 88, 176],
        [6, 8, 1, 3, 4, 16, 24, 44, 88, 156],
        [7, 1, 4, 4, 6, 10, 24, 44, 78, 156],
        [7, 2, 8, 4, 4, 16, 24, 44, 88, 156],
        [7, 3, 6, 4, 4, 8, 22, 44, 78, 152],
        [7, 4, 2, 2, 8, 12, 22, 44, 78, 156],
        [7, 4, 4, 8, 6, 6, 24, 44, 80, 154],
        [7, 6, 2, 4, 4, 16, 24, 44, 88, 156],
        [7, 8, 2, 2, 4, 16, 22, 44, 86, 152],
        [7, 8, 4, 4, 6, 10, 24, 44, 78, 156],
        [7, 9, 8, 8, 2, 2, 28, 48, 88, 176],
        [8, 4, 4, 1, 1, 14, 24, 44, 88, 156],
        [8, 4, 4, 6, 1, 9, 24, 44, 78, 156],
        [9, 3, 4, 8, 2, 6, 32, 48, 88, 176],
        [9, 5, 2, 4, 4, 8, 32, 44, 88, 172],
        [9, 8, 4, 4, 2, 14, 24, 44, 88, 156],
    ]

    total_seeds = len(all_seeds)
    logging.info(f"Starting parallel analysis for {total_seeds} seeds")
    logging.info(f"Max n: {max_n}, Optimizer: {optimizer.__name__}")
    logging.info(f"SSD check: {'enabled' if check_ssd else 'disabled'}")
    logging.info(f"Using {num_workers or 'auto'} worker(s)")

    # Prepare work items
    work_items = [(seed, max_n, optimizer, d_seed_reference, check_ssd) for seed in all_seeds]

    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(_worker_wrapper, work_items)

    # Print summary
    logging.info("\n" + "=" * 50)
    logging.info("SUMMARY")
    logging.info("=" * 50)

    successful = sum(1 for _, met_condition in results if met_condition)
    logging.info(f"Seeds meeting condition: {successful}/{total_seeds}")

    if successful > 0:
        logging.info("\nSeeds that met the condition:")
        for seed, met_condition in results:
            if met_condition:
                logging.info(f"  ✓ {seed}")

    logging.info(f"\nTotal processed: {total_seeds}")
    logging.info(f"Condition met: {successful}")
    logging.info(f"Condition not met: {total_seeds - successful}")
