#!/usr/bin/env python3
"""
submit_downloads.py
Submit one LSF job per peak file + one job for all PWMs.

Usage
-----
    python submit_downloads.py --meta metadata/big_subset.csv [options]

Options
-------
  --meta        PATH   Path to metadata CSV                            [required]
  --peaks_dir   DIR    Output directory for peaks                      [peaks_all]
  --pwms_dir    DIR    Output directory for PWMs                       [pwm_all]
  --log_dir     DIR    Directory for LSF stdout/stderr logs            [lsf_logs/downloads]
  --queue       STR    LSF queue name                                  [short]
  --memory      STR    LSF memory limit (MB)                           [2000]
  --python      STR    Python executable on the cluster                [python]
  --script      STR    Path to download_one.py                         [./download_one.py]
  --dry_run            Print bsub commands without submitting
"""

import argparse
import os
import subprocess

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--meta',       required=True)
    p.add_argument('--peaks_dir',  default='peaks_all')
    p.add_argument('--pwms_dir',   default='pwm_all')
    p.add_argument('--log_dir',    default='lsf_logs/downloads')
    p.add_argument('--queue',      default='short')
    p.add_argument('--memory',     default='2000')
    p.add_argument('--python',     default='python')
    p.add_argument('--script',     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_one.py'))
    p.add_argument('--dry_run',    action='store_true')
    return p.parse_args()


def bsub_cmd(job_name, log_out, log_err, queue, memory, command):
    return (
        f'bsub '
        f'-J {job_name} '
        f'-q {queue} '
        f'-n 1 '
        f'-R "span[hosts=1] rusage[mem={memory}]" '
        f'-o {log_out} '
        f'-e {log_err} '
        f'"{command}"'
    )


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    df = pd.read_csv(args.meta)

    # ── Peak jobs (one per row) ───────────────────────────────────────────────
    missing_href = df['peak_href'].isna() | (df['peak_href'] == '')
    if missing_href.any():
        print(f'WARNING: {missing_href.sum()} rows have no peak_href — skipping.')
    rows = df[~missing_href].reset_index(drop=True)

    n_submitted = n_skipped = 0
    for idx, row in rows.iterrows():
        cl   = row['cell_line']
        gene = row['gene_symbol']
        out_path = os.path.join(args.peaks_dir, cl, f'{gene}.bed')

        if os.path.exists(out_path):
            n_skipped += 1
            continue

        job_name = f'dl_peak_{cl}_{gene}'
        log_out  = os.path.join(args.log_dir, f'{job_name}.out')
        log_err  = os.path.join(args.log_dir, f'{job_name}.err')
        command  = (f'{args.python} {args.script} '
                    f'--meta {args.meta} --idx {idx} --out_dir {args.peaks_dir}')
        cmd = bsub_cmd(job_name, log_out, log_err, args.queue, args.memory, command)

        if args.dry_run:
            print(cmd)
        else:
            subprocess.run(cmd, shell=True, check=True)
        n_submitted += 1

    action = 'Would submit' if args.dry_run else 'Submitted'
    print(f'Peaks: {action} {n_submitted} jobs  |  skipped {n_skipped} (already exist)  '
          f'|  total rows = {len(rows)}')

    # ── PWM job (one job for all unique motifs) ───────────────────────────────
    job_name = 'dl_pwms'
    log_out  = os.path.join(args.log_dir, f'{job_name}.out')
    log_err  = os.path.join(args.log_dir, f'{job_name}.err')
    command  = (f'{args.python} {args.script} '
                f'--meta {args.meta} --pwms --out_dir {args.pwms_dir}')
    cmd = bsub_cmd(job_name, log_out, log_err, args.queue, args.memory, command)

    if args.dry_run:
        print(cmd)
    else:
        subprocess.run(cmd, shell=True, check=True)
    print(f'PWMs: {action} 1 job')


if __name__ == '__main__':
    main()
