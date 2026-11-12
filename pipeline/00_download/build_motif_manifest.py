"""TF-MoDISco reports and Fi-NeMo motif instances for the ALL_BPNET experiments, as PAPER has them: the
'sequence motifs report' (tar) and 'sequence motifs instances' (tar + bigBed) files of the same BPNet-model annotation
already chosen in all_bpnet_meta.csv (bpnet_model_annotation; same status as the model, released or archived).
Adds modisco_report_accession, motif_instances_accession, motif_instances_bigbed_accession to all_bpnet_meta.csv and
writes manifest_motifs.json (download: download_all.py manifest_motifs.json). Bulk ENCODE searches cached in metadata/."""
import json, re, time
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths

BASE = paths.ALL_BPNET
CACHE = BASE / 'metadata'


def get(path):  # as in build_all_table.py (not imported: that module runs on import)
    dest = CACHE / (re.sub(r'[^A-Za-z0-9_.-]', '_', path.strip('/'))[:200] + '.json')
    if dest.exists():
        return json.loads(dest.read_text())
    for attempt in range(4):
        try:
            with urlopen(Request('https://www.encodeproject.org' + path, headers={'Accept': 'application/json'}), timeout=600) as r:
                data = json.load(r)
            dest.write_text(json.dumps(data))
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10)


def search(q):
    return get(f'/search/?{q}&status=released&status=archived&frame=object&limit=all')['@graph']


def acc(path):
    return path.strip('/').split('/')[-1]


files = {}
for ot in ['sequence motifs report', 'sequence motifs instances']:
    for f in search('type=File&output_type=' + ot.replace(' ', '+')):
        files.setdefault(acc(f['dataset']), []).append(f)

meta = pd.read_csv(paths.META_ALL, keep_default_na=False)
manifest, cols = {}, {'modisco_report_accession': [], 'motif_instances_accession': [], 'motif_instances_bigbed_accession': []}
for r in meta.itertuples():
    mf = [f for a in r.bpnet_model_annotation.split(';') for f in files.get(a, []) if f['status'] == r.bpnet_model_status]
    pick = lambda ot, fmt=None: sorted((f for f in mf if f['output_type'] == ot and fmt in (None, f['file_format'])),
                                       key=lambda f: f['accession'])
    rep, hits, bb = pick('sequence motifs report'), pick('sequence motifs instances', 'tar'), pick('sequence motifs instances', 'bigBed')
    for c, fs in zip(cols, [rep, hits, bb]):
        cols[c].append(';'.join(f['accession'] for f in fs))
    for f in rep + hits + bb:
        manifest[f['accession']] = {'href': f['href'], 'md5': f['md5sum'], 'size': f.get('file_size', 0),
                                    'output_type': f['output_type'], 'dest': r.dir}
for c, v in cols.items():
    meta[c] = v
meta.to_csv(paths.META_ALL, index=False)
(BASE / 'manifest_motifs.json').write_text(json.dumps(manifest, indent=1))

for c in cols:
    print(f'{c}: missing {(meta[c] == "").sum()} | multiple {meta[c].str.contains(";").sum()}')
print('missing by model status:', meta[meta.modisco_report_accession == ''].bpnet_model_status.value_counts().to_dict(),
      '| no bigBed:', meta[meta.motif_instances_bigbed_accession == ''].bpnet_model_status.value_counts().to_dict())
tot = pd.DataFrame.from_dict(manifest, orient='index')
print(tot.groupby('output_type')['size'].agg(files='size', GB=lambda s: s.sum() / 1e9).round(1).to_string())
print('TOTAL GB', round(tot['size'].sum() / 1e9, 1), '| files', len(tot))
print('DONE', flush=True)
