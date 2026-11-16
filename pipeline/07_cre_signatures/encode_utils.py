"""encode_utils.py — ENCODE API query and file extraction utilities.

Edit this file directly; re-run the import cell in the notebook to pick up changes
(no kernel restart or notebook reload needed).
"""
import gzip
import os
import requests # type: ignore
import shutil
import time
import pandas as pd # type: ignore

ENCODE_BASE = 'https://www.encodeproject.org'
HEADERS = {'Accept': 'application/json'}


def query_encode_files(cell_line, output_type, file_format=None, retries=3, delay=2):
    """Query ENCODE search API for TF ChIP-seq files."""
    params = {
        'type': 'File',
        'assay_title': 'TF ChIP-seq',
        'output_type': output_type,
        'biosample_ontology.term_name': cell_line,
        'status': 'released',
        'assembly': 'GRCh38',
        'format': 'json',
        'limit': 'all',
        'frame': 'embedded',
    }
    if file_format is not None:
        params['file_format'] = file_format
    url = f'{ENCODE_BASE}/search/'
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=120)
            resp.raise_for_status()
            return resp.json().get('@graph', [])
        except requests.RequestException as e:
            print(f'  Attempt {attempt + 1} failed for {cell_line} / {output_type}: {e}')
            if attempt < retries - 1:
                time.sleep(delay)
    return []


def _looks_like_gene(s):
    """Return True if string looks like a gene symbol, not a URL path."""
    return bool(s) and not s.startswith('/')


def extract_file_info(file_obj, cell_line):
    """Extract relevant fields from an ENCODE file object."""
    # --- Target / TF name ---
    target = file_obj.get('target', {})
    if isinstance(target, dict):
        tf_name = target.get('label', target.get('name', ''))
    elif isinstance(target, str):
        tf_name = target
    else:
        tf_name = ''

    # --- Gene symbol ---
    gene_symbol = ''
    if isinstance(target, dict):
        genes = target.get('genes', [])
        if genes and isinstance(genes[0], dict):
            gene_symbol = genes[0].get('symbol', '')
        elif genes and isinstance(genes[0], str) and _looks_like_gene(genes[0]):
            gene_symbol = genes[0]
    if not _looks_like_gene(gene_symbol):
        gene_symbol = tf_name

    # --- Experiment accession ---
    dataset = file_obj.get('dataset', '')
    if isinstance(dataset, dict):
        exp_acc = dataset.get('accession', '')
    else:
        parts = str(dataset).strip('/').split('/')
        exp_acc = parts[-1] if parts else ''

    # --- File size ---
    file_size_mb = round((file_obj.get('file_size', 0) or 0) / 1e6, 2)

    return {
        'tf_name':               tf_name,
        'gene_symbol':           gene_symbol,
        'cell_line':             cell_line,
        'experiment_accession':  exp_acc,
        'file_accession':        file_obj.get('accession', ''),
        'href':                  file_obj.get('href', ''),
        'biological_replicates': file_obj.get('biological_replicates', []),
        'file_size_mb':          file_size_mb,
        'file_format':           file_obj.get('file_format', ''),
        'output_type':           file_obj.get('output_type', ''),
    }


def select_best_file(group):
    """Select the best representative file from a group belonging to one experiment.

    Preference: pooled replicates (len(bio_reps) > 1), then most replicates, then first.
    """
    if len(group) == 1:
        return group.iloc[[0]]
    group = group.copy()
    group['n_reps'] = group['biological_replicates'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    pooled = group[group['n_reps'] > 1]
    if len(pooled) > 0:
        return pooled.sort_values('n_reps', ascending=False).iloc[[0]]
    return group.sort_values('n_reps', ascending=False).iloc[[0]]


def _groupby_apply(df, keys, func):
    """groupby + apply compatible with pandas <2.2 and >=2.2.

    Returns a flat DataFrame with group keys restored as columns.
    """
    grouped = df.groupby(keys)
    try:
        result = grouped.apply(func, include_groups=False)
    except TypeError:
        result = grouped.apply(func)
    result = result.reset_index(level=list(range(len(keys)))).reset_index(drop=True)
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def download_peaks(df, out_dir='peaks', dry_run=True):
    """Download conservative IDR peak BED files, one folder per cell line.

    Files are saved as {out_dir}/{cell_line}/{gene_symbol}.bed (decompressed).
    Skips files that already exist.

    Parameters
    ----------
    df       : DataFrame with columns gene_symbol, cell_line, peak_href
    out_dir  : root output directory (default 'peaks')
    dry_run  : if True, print what would be downloaded without downloading
    """
    if dry_run:
        print('=== DRY RUN — no files will be downloaded ===')

    missing_href = df['peak_href'].isna() | (df['peak_href'] == '')
    if missing_href.any():
        print(f'WARNING: {missing_href.sum()} rows have no peak_href — skipping.')

    rows = df[~missing_href].copy()
    total = len(rows)
    downloaded = skipped = failed = 0

    for i, row in enumerate(rows.itertuples(), 1):
        cell_line  = row.cell_line
        gene       = row.gene_symbol
        href       = row.peak_href

        cell_dir = os.path.join(out_dir, cell_line)
        out_path = os.path.join(cell_dir, f'{gene}.bed')

        if os.path.exists(out_path):
            print(f'[{i}/{total}] SKIP  {cell_line}/{gene}.bed (exists)')
            skipped += 1
            continue

        url = ENCODE_BASE + href if href.startswith('/') else href
        print(f'[{i}/{total}] {"(dry) " if dry_run else ""}GET   {cell_line}/{gene}.bed  ← {url}')

        if dry_run:
            continue

        os.makedirs(cell_dir, exist_ok=True)
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                # Decompress .gz on the fly; plain .bed passes through
                if href.endswith('.gz'):
                    tmp = out_path + '.gz'
                    with open(tmp, 'wb') as fh:
                        shutil.copyfileobj(resp.raw, fh)
                    with gzip.open(tmp, 'rb') as gz_in, open(out_path, 'wb') as bed_out:
                        shutil.copyfileobj(gz_in, bed_out)
                    os.remove(tmp)
                else:
                    with open(out_path, 'wb') as fh:
                        shutil.copyfileobj(resp.raw, fh)
            downloaded += 1
        except Exception as e:
            print(f'  ERROR: {e}')
            failed += 1

    print(f'\nDone. downloaded={downloaded}  skipped={skipped}  failed={failed}  total={total}')
    if dry_run:
        print('Set dry_run=False to actually download.')


HOCOMOCO_PWM_BASE = (
    'https://hocomoco14.autosome.org/final_bundle/hocomoco14'
    '/H14CORE-CLUSTERED/pwm'
)


def download_motifs(df, out_dir='pwm', dry_run=True):
    """Download HOCOMOCO14 PWM files — one file per unique representative motif.

    Files are saved as {out_dir}/{motif_id}.pwm.
    Deduplicates by representative_mot_id so each cluster motif is downloaded once.
    Skips files that already exist.

    Parameters
    ----------
    df      : DataFrame with column representative_mot_id
    out_dir : root output directory (default 'pwm')
    dry_run : if True, print what would be downloaded without downloading
    """
    if dry_run:
        print('=== DRY RUN — no files will be downloaded ===')

    if 'representative_mot_id' not in df.columns:
        raise ValueError("Missing column: 'representative_mot_id'")

    unique_motifs = (
        df['representative_mot_id']
        .dropna()
        .loc[lambda s: s != '']
        .unique()
        .tolist()
    )
    total = len(unique_motifs)
    print(f'Unique motifs to download: {total}')
    downloaded = skipped = failed = 0

    os.makedirs(out_dir, exist_ok=True)

    for i, motif_id in enumerate(unique_motifs, 1):
        out_path = os.path.join(out_dir, f'{motif_id}.pwm')

        if os.path.exists(out_path):
            print(f'[{i}/{total}] SKIP  {motif_id}.pwm (exists)')
            skipped += 1
            continue

        url = f'{HOCOMOCO_PWM_BASE}/{motif_id}.pwm'
        print(f'[{i}/{total}] {"(dry) " if dry_run else ""}GET   {motif_id}.pwm')

        if dry_run:
            continue

        try:
            with requests.get(url, headers=HEADERS, timeout=30) as resp:
                resp.raise_for_status()
                with open(out_path, 'w') as fh:
                    fh.write(resp.text)
            downloaded += 1
        except Exception as e:
            print(f'  ERROR: {e}')
            failed += 1

    print(f'\nDone. downloaded={downloaded}  skipped={skipped}  failed={failed}  total={total}')
    if dry_run:
        print('Set dry_run=False to actually download.')
