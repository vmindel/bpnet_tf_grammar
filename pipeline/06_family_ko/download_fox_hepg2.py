"""Download the BPNet observed-signal (count) bigWigs, plus and minus strand, for every released
FOX-family BPNet model trained in HepG2. One folder per model, {TF}_HepG2_{BPNet_accession}/, using the
same file naming as klf_sp/. Also writes fox_hepg2_bpnet_accessions.tsv. Resumable (md5-checked)."""
import csv, hashlib, json, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
BASE = paths.ROOT / 'fox_hepg2'
BASE.mkdir(parents=True, exist_ok=True)
ENC = 'https://www.encodeproject.org'
HEPG2 = '/biosample-types/cell_line_EFO_0001187/'
OUTPUT_TYPES = {'observed signal profile (plus strand)': 'observed_signal_profile_plus',
                'observed signal profile (minus strand)': 'observed_signal_profile_minus'}


def get(path):
    url = ENC + path + ('&' if '?' in path else '?') + 'format=json'
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'Accept': 'application/json'}), timeout=120) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(f, out):
    if out.exists() and md5(out) == f['md5sum']:
        return out.name, 'already present'
    for attempt in range(3):
        subprocess.run(['curl', '-sSL', '--retry', '5', '-C', '-', '-o', str(out), ENC + f['href']], check=False)
        if out.exists() and md5(out) == f['md5sum']:
            return out.name, 'downloaded'
        out.unlink(missing_ok=True)
    return out.name, 'FAILED md5/download'


models = get('/search/?type=Annotation&annotation_type=BPNet-model&status=released&frame=object&limit=all')['@graph']
rows, jobs = [], []
for m in models:
    tfs = [t.strip('/').split('/')[-1].replace('-human', '') for t in m['targets']]
    if m.get('biosample_ontology') != HEPG2 or not any(t.startswith('FOX') for t in tfs):
        continue
    files = get(f"/search/?type=File&dataset=/annotations/{m['accession']}/&status=released"
                '&file_format=bigWig&frame=object&limit=all')['@graph']
    d = BASE / f"{','.join(tfs)}_HepG2_{m['accession']}"
    d.mkdir(exist_ok=True)
    row = {'CellLine': 'HepG2', 'TF': ','.join(tfs), 'BPNet_accession': m['accession'],
           'Source_ChIPseq_experiment': m['experimental_input'][0].strip('/').split('/')[-1],
           'Model_QC': re.sub(r'^.*?\(ENCSR\w+\)', '', m.get('description', ''))}
    for ot, slug in OUTPUT_TYPES.items():
        hits = [f for f in files if f['output_type'] == ot]
        row[f"ObservedSignal_{slug.rsplit('_', 1)[1]}_bigWig"] = ';'.join(f['accession'] for f in hits)
        jobs += [(f, d / f"{slug}__{f['accession']}.bigWig") for f in hits]
    print('MODEL', row, flush=True)
    rows.append(row)

rows.sort(key=lambda r: (r['TF'], r['BPNet_accession']))
with open(paths.METADATA / 'fox_hepg2_bpnet_accessions.tsv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['CellLine', 'TF', 'BPNet_accession', 'ObservedSignal_plus_bigWig',
                                       'ObservedSignal_minus_bigWig', 'Source_ChIPseq_experiment', 'Model_QC'],
                       delimiter='\t')
    w.writeheader()
    w.writerows(rows)
print('models', len(rows), 'files to fetch', len(jobs), flush=True)

failed = []
with ThreadPoolExecutor(6) as ex:
    for i, fu in enumerate(as_completed([ex.submit(fetch, f, out) for f, out in jobs]), 1):
        name, status = fu.result()
        print(i, name, status, flush=True)
        if status.startswith('FAILED'):
            failed.append(name)
print('FAILED', failed, flush=True)
print('DONE', flush=True)
