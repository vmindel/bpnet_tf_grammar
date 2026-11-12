"""Download files listed in klf_sp_bpnet_accessions.tsv (all but fold-change bigWigs) into
per-model folders {TF}_{CellLine}_{BPNetDataset}/. Controls go to _shared/ and are symlinked.
Tars are extracted into {output_type}__{accession}/ and the archive deleted. Resumable."""
import csv, json, glob, hashlib, re, subprocess, tarfile, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
BASE = paths.ROOT / 'klf_sp'
SHARED = BASE / '_shared'
SHARED.mkdir(parents=True, exist_ok=True)
ENC = 'https://www.encodeproject.org'
COLS = ['NegativeBigWig_plus_strand_accession', 'NegativeBigWig_minus_strand_accession', 'Peaks_accession',
        'MotifReport_accession', 'Models_accession', 'Motifs_accession', 'MotifInstances_accession',
        'MotifInstances_bigBed', 'ObservedSignal_plus_bigWig', 'ObservedSignal_minus_bigWig']
CONTROL_COLS = COLS[:2]

meta = {}
for f in glob.glob(str(BASE / 'metadata' / 'search*.json')):
    for r in json.loads(Path(f).read_text()).get('@graph', []):
        if 'accession' in r and 'md5sum' in r:
            meta[r['accession']] = r


def local_name(acc):
    m = meta[acc]
    ot = m['output_type']
    strand = '_plus' if 'plus strand' in ot else '_minus' if 'minus strand' in ot else ''
    slug = re.sub(r'[^a-z0-9]+', '_', re.sub(r'\((plus|minus) strand\)', '', ot.lower())).strip('_')
    if 'IDR' in ot:
        slug = 'chipseq_' + slug
    ext = m['href'].rsplit('/', 1)[-1].split('.', 1)[1]  # bigWig / bed.gz / tar.gz
    return f'{slug}{strand}__{acc}', ext


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(acc, dest_dir):
    stem, ext = local_name(acc)
    is_tar = ext.startswith('tar')
    out = dest_dir / f'{stem}.{ext}'
    xdir = dest_dir / stem
    if is_tar and xdir.is_dir() and any(xdir.iterdir()) and not out.exists():
        return acc, 'already extracted'
    if not (out.exists() and md5(out) == meta[acc]['md5sum']):
        for attempt in range(3):
            subprocess.run(['curl', '-sSL', '--retry', '5', '-C', '-', '-o', str(out), ENC + meta[acc]['href']],
                           check=False)
            if out.exists() and md5(out) == meta[acc]['md5sum']:
                break
            out.unlink(missing_ok=True)
        else:
            return acc, 'FAILED md5/download'
    if is_tar:
        xdir.mkdir(exist_ok=True)
        with tarfile.open(out, 'r:*') as t:
            t.extractall(xdir, filter='data')
        if any(xdir.iterdir()):
            out.unlink()
        return acc, 'downloaded+extracted'
    return acc, 'downloaded'


rows = list(csv.DictReader(open(BASE / 'klf_sp_bpnet_accessions.tsv'), delimiter='\t'))
jobs = {}  # (acc, dir) -> None, dedup shared controls
links = []
for r in rows:
    d = BASE / f"{r['TF']}_{r['CellLine'].replace(' ', '_')}_{r['BPNet_accession']}"
    d.mkdir(exist_ok=True)
    for c in COLS:
        for acc in filter(None, r[c].split(';')):
            if c in CONTROL_COLS:
                jobs[(acc, SHARED)] = None
                links.append((acc, d))
            else:
                jobs[(acc, d)] = None
print('files to fetch', len(jobs), flush=True)

failed = []
with ThreadPoolExecutor(6) as ex:
    futs = [ex.submit(fetch, a, d) for a, d in jobs]
    for i, fu in enumerate(as_completed(futs), 1):
        acc, status = fu.result()
        print(i, acc, status, flush=True)
        if status.startswith('FAILED'):
            failed.append(acc)

for acc, d in links:
    stem, ext = local_name(acc)
    link = d / f'{stem}.{ext}'
    if not link.exists():
        os.symlink(os.path.relpath(SHARED / f'{stem}.{ext}', d), link)
print('FAILED', failed, flush=True)
print('DONE', flush=True)
