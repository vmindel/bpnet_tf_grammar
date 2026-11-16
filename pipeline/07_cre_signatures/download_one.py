#!/usr/bin/env python3
"""
download_one.py
Download a single peak BED file (by row index) or all PWMs from metadata.

Usage
-----
    # Download peak for row 42:
    python download_one.py --meta metadata/big_subset.csv --idx 42 --out_dir peaks_all

    # Download all unique PWMs:
    python download_one.py --meta metadata/big_subset.csv --pwms --out_dir pwm_all
"""

import argparse
import importlib
import os
import sys

# encode_utils.py sits next to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encode_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--meta',    required=True, help='Path to metadata CSV')
    p.add_argument('--out_dir', required=True, help='Output directory')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--idx',  type=int, help='Row index for peak download')
    group.add_argument('--pwms', action='store_true', help='Download all unique PWMs')
    return p.parse_args()


def main():
    import pandas as pd
    args = parse_args()

    df = pd.read_csv(args.meta)

    if args.pwms:
        encode_utils.download_motifs(df, out_dir=args.out_dir, dry_run=False)
    else:
        row = df.iloc[[args.idx]]
        encode_utils.download_peaks(row, out_dir=args.out_dir, dry_run=False)


if __name__ == '__main__':
    main()
