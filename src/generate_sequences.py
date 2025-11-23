"""
Generate Motzkin–greedy difference sequences d(n) and corresponding P_n
for arbitrary seeds, optionally checking the DSS property.

Usage example:

    python src/generate_sequences.py --seed 2,1 --max-n 20 --check-ssd

Outputs:
    - d-sequence printed to stdout
    - per-n P_n and SSD flags printed to stdout
    - optional dump files in a chosen directory
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from motzkin_greedy import motzkin_greedy
from utils import P_from_d_prefix, parse_seed
from ssd_check import (
    is_distinct_subset_sum_combinations,
    is_distinct_subset_sum_bitmask,
)


def generate_for_seed(seed: str, max_n: int, out_dir: Path, check_ssd: bool) -> None:
    d1, d2 = parse_seed(seed)
    print(f"Seed (d1,d2) = ({d1},{d2}), max_n = {max_n}")

    d_seq = motzkin_greedy(max_n, d1, d2)
    print("\nDifference sequence d(1..n):")
    print(d_seq)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save d-sequence
    d_file = out_dir / f"seed_{d1}_{d2}_d.csv"
    with d_file.open("w") as f:
        f.write(",".join(str(x) for x in d_seq) + "\n")
    print(f"\nSaved d-sequence to: {d_file}")

    # Build P_n and optionally check SSD
    P_data: Dict[int, Dict[str, Any]] = {}
    ssd_flags: List[int] = []
    max_P_list: List[int] = []

    print("\nPer-n data:")
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
                print(f"WARNING: SSD checkers disagree at n={n}: comb={ok_c}, mask={ok_b}")
            ok = ok_c and ok_b
            ssd_flags.append(1 if ok else 0)
            record["ssd"] = ok
            if not ok:
                record["collision_combinations"] = info_c
                record["collision_bitmask"] = info_b
                print(f"n={n}: NOT DSS, collision (combinations): {info_c}")
            else:
                print(f"n={n}: SSD, maxP={maxP}")
        else:
            print(f"n={n}: maxP={maxP}")

        P_data[n] = record

    # Save P data as JSON
    P_file = out_dir / f"P_sets_seed_{d1}_{d2}.json"
    with P_file.open("w") as f:
        json.dump(P_data, f, indent=2)
    print(f"\nSaved P_n data to: {P_file}")

    # Save summary table
    summary_file = out_dir / f"summary_seed_{d1}_{d2}.csv"
    with summary_file.open("w") as f:
        f.write("n,d_n,maxP,SSD_flag\n")
        for n in range(1, max_n + 1):
            d_n = d_seq[n - 1]
            maxP = max_P_list[n - 1]
            ssd_flag = ssd_flags[n - 1] if check_ssd else -1
            f.write(f"{n},{d_n},{maxP},{ssd_flag}\n")
    print(f"Saved summary to: {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Motzkin–greedy d(n) and P_n for a given seed.")
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Seed as 'd1,d2', e.g. '2,1' or '1,1'.",
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

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    generate_for_seed(args.seed, args.max_n, out_dir, args.check_ssd)


if __name__ == "__main__":
    main()
