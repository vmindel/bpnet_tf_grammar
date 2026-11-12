"""Assemble ENCODE BPNet model / control / peak accessions for KLF/SP family TFs.
Output: klf_sp_bpnet_accessions.tsv (one row per BPNet model annotation)."""
import json, re, csv, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
BASE = paths.ROOT / 'klf_sp'
CACHE = BASE / 'metadata'
CACHE.mkdir(parents=True, exist_ok=True)
ENC = 'https://www.encodeproject.org'


def get(path):
    url = ENC + path + ('&' if '?' in path else '?') + 'format=json'
    dest = CACHE / (re.sub(r'[^A-Za-z0-9_.-]', '_', path.strip('/'))[:200] + '.json')
    if dest.exists():
        return json.loads(dest.read_text())
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'Accept': 'application/json'}), timeout=120) as r:
                data = json.load(r)
            dest.write_text(json.dumps(data))
            return data
        except HTTPError as e:
            if e.code == 404 and path.startswith('/search/'):  # ENCODE search: 404 == zero hits
                return {'@graph': []}
            if attempt == 3:
                raise
            time.sleep(3)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3)


def acc(path):
    return path.strip('/').split('/')[-1]


def dataset_files(ds, kind):
    return get(f'/search/?type=File&dataset=/{kind}/{ds}/&status=released&frame=object&limit=all')['@graph']


def pick(files, output_type, fmt=None):
    hits = [f for f in files if f['output_type'] == output_type and (fmt is None or f['file_format'] == fmt)]
    return ';'.join(sorted(f['accession'] for f in hits))


models = get('/search/?type=Annotation&annotation_type=BPNet-model&status=released&frame=object&limit=all')['@graph']
controls = get('/search/?type=Annotation&annotation_type=BPNet-control&status=released&frame=object&limit=all')['@graph']
ctl_by_inputs = {}
for c in controls:
    ctl_by_inputs.setdefault(frozenset(acc(x) for x in c['experimental_input']), []).append(c)

fam = re.compile(r'(KLF\d+|SP\d+)')
rows = []
for m in models:
    tfs = [acc(t).replace('-human', '') for t in m['targets']]
    if not any(fam.fullmatch(t) for t in tfs):
        continue
    src = acc(m['experimental_input'][0])
    print('MODEL', m['accession'], tfs, src, flush=True)
    mf = dataset_files(m['accession'], 'annotations')
    expd = get(f'/experiments/{src}/')
    cell = expd['biosample_ontology']['term_name']
    ctl_exps = frozenset(acc(x['@id'] if isinstance(x, dict) else x) for x in expd.get('possible_controls', []))

    # BPNet-control annotation: exact match on control experiment set, else best overlap
    ctl = ctl_by_inputs.get(ctl_exps, [])
    match = 'exact'
    if not ctl:
        scored = sorted(((len(k & ctl_exps), v) for k, v in ctl_by_inputs.items() if k & ctl_exps),
                        key=lambda x: -x[0])
        ctl = scored[0][1] if scored else []
        match = 'partial' if ctl else 'none'
    ctl_plus = ctl_minus = ctl_acc = ''
    if ctl:
        cf = dataset_files(ctl[0]['accession'], 'annotations')
        ctl_acc = ';'.join(c['accession'] for c in ctl)
        ctl_plus = ';'.join(f['accession'] for f in cf if 'plus' in f['output_type'] and f['file_format'] == 'bigWig')
        ctl_minus = ';'.join(f['accession'] for f in cf if 'minus' in f['output_type'] and f['file_format'] == 'bigWig')
        print('  CTL', ctl_acc, [(f['accession'], f['output_type']) for f in cf], flush=True)

    # source ChIP-seq: IDR optimal peaks + pooled FC bigWig from the current analysis
    current = [a for a in expd.get('analyses', []) if a['status'] == 'released' and a.get('assembly') == 'GRCh38']
    current_files = {acc(f) for a in current for f in a['files']}
    # peak fallback: optimal IDR -> conservative IDR -> pooled-replicate IDR thresholded
    beds = get(f'/search/?type=File&dataset=/experiments/{src}/&file_format=bed&file_format_type=narrowPeak'
               '&assembly=GRCh38&status=released&frame=object&limit=all')['@graph']
    beds = [p for p in beds if p['accession'] in current_files] or beds
    peaks_cur = [p for p in beds if p.get('preferred_default')]  # ENCODE's default (optimal-equivalent) set
    peak_type = ';'.join(sorted({p['output_type'] + ' (preferred_default)' for p in peaks_cur}))
    for ot in [] if peaks_cur else ['optimal IDR thresholded peaks', 'conservative IDR thresholded peaks', 'IDR thresholded peaks']:
        hits = [p for p in beds if p['output_type'] == ot]
        if ot == 'IDR thresholded peaks' and hits:
            nrep = max(len(p.get('biological_replicates', [])) for p in hits)
            hits = [p for p in hits if len(p.get('biological_replicates', [])) == nrep]
        if hits:
            peaks_cur, peak_type = hits, ot
            break
    fc = get(f'/search/?type=File&dataset=/experiments/{src}/&output_type=fold+change+over+control'
             '&file_format=bigWig&assembly=GRCh38&status=released&frame=object&limit=all')['@graph']
    fc_cur = [r for r in fc if r['accession'] in current_files] or fc
    fc_best = max(fc_cur, key=lambda r: len(r.get('biological_replicates', [])))['accession'] if fc_cur else ''

    rows.append({
        'CellLine': cell,
        'TF': ','.join(tfs),
        'BPNet_accession': m['accession'],
        'NegativeBigWig_plus_strand_accession': ctl_plus,
        'NegativeBigWig_minus_strand_accession': ctl_minus,
        'Peaks_accession': ';'.join(p['accession'] for p in peaks_cur),
        'Peaks_type': peak_type,
        'MotifReport_accession': pick(mf, 'sequence motifs report'),
        'Models_accession': pick(mf, 'models'),
        'Motifs_accession': pick(mf, 'sequence motifs'),
        'MotifInstances_accession': pick(mf, 'sequence motifs instances', 'tar'),
        'MotifInstances_bigBed': pick(mf, 'sequence motifs instances', 'bigBed'),
        'ObservedSignal_plus_bigWig': pick(mf, 'observed signal profile (plus strand)', 'bigWig'),
        'ObservedSignal_minus_bigWig': pick(mf, 'observed signal profile (minus strand)', 'bigWig'),
        'FoldChangeOverControl_bigWig': fc_best,
        'Source_ChIPseq_experiment': src,
        'Control_experiments': ','.join(sorted(ctl_exps)),
        'BPNet_control_dataset': ctl_acc,
        'Control_match': match,
        'Model_QC': m.get('description', '').split(f'({src})', 1)[-1],
    })

rows.sort(key=lambda r: (r['TF'].rstrip('0123456789'), int(re.sub(r'\D', '', r['TF'].split(',')[0])), r['CellLine']))
with open(BASE / 'klf_sp_bpnet_accessions.tsv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t')
    w.writeheader()
    w.writerows(rows)
print('DONE', len(rows), flush=True)
