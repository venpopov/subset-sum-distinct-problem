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


def generate_ssd_df(d_seed: List[int], max_n: int, optimizer: Callable = optimize_motzkin_greedy) -> pd.DataFrame:
    """
    Generates a greedy sequence based on a seed and checks for
    Distinct Subset Sum (SSD) properties up to max_n.
    Caches results to avoid redundant computations.

    Args:
        d_seed: Initial seed sequence
        max_n: Maximum n value to compute
        optimizer: Function to use for optimization (default: optimize_motzkin_greedy)
                   Can also pass optimize_smooth_walk_greedy
    """
    # Create a hashable key from arguments
    cache_key = (tuple(d_seed), max_n, optimizer.__name__)

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

        ok, info = is_distinct_subset_sum_optimized(Pn)
        ssd_flags.append(1 if ok else 0)

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

    Returns:
        DataFrame with renamed columns: SSD, d, P, delta (or tuple if return_for_multiprocessing=True)
    """
    try:
        if return_for_multiprocessing:
            logging.info(f"Processing seed {d_seed}")

        df_reference = generate_ssd_df(d_seed_reference, max_n)
        df = generate_ssd_df(d_seed, max_n, optimizer=optimizer)
        df = df.join(df_reference.set_index("n"), on="n", lsuffix="_new", rsuffix="_ref")
        df["maxp_delta"] = df["maxP_new"] - df["maxP_ref"]

        # Create display dataframe
        display_df = df[["SSD_flag_new", "d_n_new", "maxP_new", "maxp_delta"]].copy()
        display_df.columns = ["SSD", "d", "P", "delta"]

        # Check condition 1: rows after first two with delta <= 0 and SSD == 1
        condition_met = (display_df.iloc[2:]["delta"] <= 0) & (display_df.iloc[2:]["SSD"] == 1)

        # Check condition 2: no more than 2/3 of SSD values are 0
        ssd_zero_ratio = (display_df["SSD"] == 0).sum() / len(display_df)
        ssd_condition_met = ssd_zero_ratio <= 2 / 3

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


def _worker_wrapper(args: Tuple[List[int], int, Callable, List[int]]) -> Tuple[List[int], bool]:
    """Wrapper for multiprocessing pool."""
    d_seed, max_n, optimizer, d_seed_reference = args
    return compare_and_save(d_seed, max_n, optimizer, d_seed_reference, return_for_multiprocessing=True)


def is_valid_seed(seed: List[int]) -> bool:
    """
    Check if a seed is valid according to these rules:
    1. Cannot have more than two of the same element
    2. If an element appears twice, the duplicates must be adjacent

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

    return True


if __name__ == "__main__":
    # example use: python src/sequence_analysis.py --workers 12
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
    args = parser.parse_args()

    max_n = args.max_n
    num_workers = args.workers
    d_seed_reference = [1, 1]
    optimizer = optimize_smooth_walk_greedy

    # Generate all seed vectors: lengths 2-5, values 1-5
    all_seeds = []
    for length in range(2, 6):  # lengths 2, 3, 4, 5
        for values in product(range(1, 8), repeat=length):  # all combinations of 1-5
            seed = list(values)
            if is_valid_seed(seed):
                all_seeds.append(seed)

    total_seeds = len(all_seeds)
    logging.info(f"Starting parallel analysis for {total_seeds} seeds")
    logging.info(f"Max n: {max_n}, Optimizer: {optimizer.__name__}")
    logging.info(f"Using {num_workers or 'auto'} worker(s)")

    # Prepare work items
    work_items = [(seed, max_n, optimizer, d_seed_reference) for seed in all_seeds]

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
