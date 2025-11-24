#!/usr/bin/env python3
"""
Script to deduplicate files in data/special_seeds based on first 25 lines.
If multiple files have the same content (first 25 lines), keeps only the one with the shortest filename.
Copies unique files to data/key_sequences.
"""

import shutil
from pathlib import Path
from collections import defaultdict


def get_first_n_lines(filepath, n=25):
    """Read and return the first n lines of a file."""
    try:
        with open(filepath, "r") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)
            return "".join(lines)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def main():
    # Define directories
    source_dir = Path("data/special_seeds")
    target_dir = Path("data/key_sequences")

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified target directory: {target_dir}")

    # Get all files from source directory
    files = sorted([f for f in source_dir.iterdir() if f.is_file()])
    print(f"Found {len(files)} files in {source_dir}")

    # Group files by their first 25 lines content
    content_to_files = defaultdict(list)

    for filepath in files:
        content = get_first_n_lines(filepath, n=25)
        if content is not None:
            content_to_files[content].append(filepath)

    print(f"Found {len(content_to_files)} unique content groups")

    # For each content group, keep only the file with shortest filename
    files_to_keep = []
    duplicates_removed = 0

    for content, file_list in content_to_files.items():
        if len(file_list) > 1:
            # Sort by filename length, then alphabetically
            file_list_sorted = sorted(file_list, key=lambda f: (len(f.name), f.name))
            files_to_keep.append(file_list_sorted[0])
            duplicates_removed += len(file_list) - 1
            print(f"\nDuplicate group ({len(file_list)} files):")
            print(f"  Keeping: {file_list_sorted[0].name}")
            for dup_file in file_list_sorted[1:]:
                print(f"  Removing: {dup_file.name}")
        else:
            files_to_keep.append(file_list[0])

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total files processed: {len(files)}")
    print(f"  Unique files to keep: {len(files_to_keep)}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"{'=' * 60}\n")

    # Copy unique files to target directory
    for filepath in files_to_keep:
        target_path = target_dir / filepath.name
        shutil.copy2(filepath, target_path)
        print(f"Copied: {filepath.name}")

    print(f"\nSuccessfully copied {len(files_to_keep)} unique files to {target_dir}")


if __name__ == "__main__":
    main()
