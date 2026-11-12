"""Resolve ENCODE BPNet (GRAMMAR) accessions for every experiment in big_subset_meta.csv.
Per experiment: BPNet-model annotation -> models tar + TF-MoDISco report tar + model performance metrics tar +
Fi-NeMo motif instances (tar + bigBed) + training and test regions tar +
pooled-pseudoreplicate IDR ranked peaks; BPNet-control annotation
(the control bigWigs the models file was derived from; possible_controls matching only as fallback) -> control profile bigWigs (plus/minus strand).
Peaks = peak_file_accession already in the metadata.
Writes big_subset_meta.csv with added columns (original kept as big_subset_meta.orig.csv) and
PAPER/manifest.json listing every file to download (href, md5, size, destination)."""
import json, os, re, time, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths

BASE = paths.PAPER                      # downloads + manifests go to the data root
META = paths.META_PAPER                 # metadata tables live in the repository
CACHE = BASE / 'metadata'
CACHE.mkdir(exist_ok=True)
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


def dataset_files(ds, status='released'):
    st = '&status=released' + ('&status=archived' if status == 'archived' else '')
    return get(f'/search/?type=File&dataset=/annotations/{ds}/{st}&frame=object&limit=all')['@graph']


def pick(files, output_type, fmt=None):
    return [f for f in files if f['output_type'] == output_type and (fmt is None or f['file_format'] == fmt)]


models = get('/search/?type=Annotation&annotation_type=BPNet-model&status=released&frame=object&limit=all')['@graph']
controls = get('/search/?type=Annotation&annotation_type=BPNet-control&status=released&frame=object&limit=all')['@graph']
# ENCODE archived the models that failed their QC; used only when no released model exists (flagged in bpnet_model_status)
arch_models = get('/search/?type=Annotation&annotation_type=BPNet-model&status=archived&frame=object&limit=all')['@graph']
arch_controls = get('/search/?type=Annotation&annotation_type=BPNet-control&status=archived&frame=object&limit=all')['@graph']
model_by_exp = {}
for m in models:
    for x in m['experimental_input']:
        model_by_exp.setdefault(acc(x), []).append(m)
arch_model_by_exp = {}
for m in arch_models:
    for x in m['experimental_input']:
        arch_model_by_exp.setdefault(acc(x), []).append(m)
ctl_by_inputs, arch_ctl_by_inputs = {}, {}
for pool, cs in [(ctl_by_inputs, controls), (arch_ctl_by_inputs, controls + arch_controls)]:
    for c in cs:
        pool.setdefault(frozenset(acc(x) for x in c['experimental_input']), []).append(c)

df = pd.read_csv(META, index_col=0)
if 'bpnet_model_annotation' in df.columns:  # rerun: start from the original columns
    df = pd.read_csv(paths.META_PAPER_ORIG, index_col=0)
else:
    shutil.copy(META, paths.META_PAPER_ORIG)

manifest = {}  # accession -> {href, md5, size, dest}


def add(f, dest):
    manifest[f['accession']] = {'href': f['href'], 'md5': f['md5sum'], 'size': f.get('file_size', 0),
                                'output_type': f['output_type'], 'dest': str(dest.relative_to(BASE))}


def resolve(r):
    exp, cell, tf = r.experiment_accession, r.cell_line, r.tf_name
    out = dict(bpnet_model_status='', bpnet_model_annotation='', bpnet_models_accession='', modisco_report_accession='', model_performance_accession='',
               motif_instances_accession='', motif_instances_bigbed_accession='', training_regions_accession='', idr_ranked_peaks_accession='',
               bpnet_control_annotation='', control_plus_bigwig_accession='', control_minus_bigwig_accession='',
               control_match='none', bpnet_model_qc='')
    ms, status = model_by_exp.get(exp, []), 'released'
    if not ms:
        ms, status = arch_model_by_exp.get(exp, []), 'archived'
    if not ms:
        return exp, out, []
    files = []
    mf = [f for m in ms for f in dataset_files(m['accession'], status)]
    mod = pick(mf, 'models')
    rep = pick(mf, 'sequence motifs report')
    perf = pick(mf, 'model performance metrics')
    hits, hits_bb = pick(mf, 'sequence motifs instances', 'tar'), pick(mf, 'sequence motifs instances', 'bigBed')
    regions = pick(mf, 'training and test regions')
    files += [(f, Path(cell) / tf) for f in mod + rep + perf + hits + hits_bb + regions]
    out.update(bpnet_model_status=status, bpnet_model_annotation=';'.join(m['accession'] for m in ms),
               bpnet_models_accession=';'.join(sorted(f['accession'] for f in mod)),
               modisco_report_accession=';'.join(sorted(f['accession'] for f in rep)),
               model_performance_accession=';'.join(sorted(f['accession'] for f in perf)),
               motif_instances_accession=';'.join(sorted(f['accession'] for f in hits)),
               motif_instances_bigbed_accession=';'.join(sorted(f['accession'] for f in hits_bb)),
               training_regions_accession=';'.join(sorted(f['accession'] for f in regions)),
               bpnet_model_qc=';'.join(m.get('description', '').split(f'({exp})', 1)[-1] for m in ms))

    expd = get(f'/experiments/{exp}/')
    # background regions overlapping these were dropped by ENCODE for AUROC/AUPRC (checked against auprc_baseline)
    ranked = [f for f in expd['files'] if f['output_type'] == 'IDR ranked peaks' and f.get('assembly') == 'GRCh38'
              and f['file_format'] == 'bed' and f['status'] == 'released'
              and 'pooled-pseudoreplicated' in ((f.get('analysis_step_version') or {}).get('analysis_step') or {}).get('name', '')]
    files += [(f, Path(cell) / tf) for f in ranked]
    out.update(idr_ranked_peaks_accession=';'.join(sorted(f['accession'] for f in ranked)))
    # control = the bigWigs the models file was derived from, i.e. the ones the model was trained with
    trained = [get(f'/files/{acc(a)}/') for m in mod for a in m.get('derived_from', [])]
    trained = [f for f in trained if f['output_type'].startswith('observed control profile')]
    if trained:
        plus = [f for f in trained if 'plus' in f['output_type']]
        minus = [f for f in trained if 'minus' in f['output_type']]
        files += [(f, Path(cell) / 'Control') for f in plus + minus]
        out.update(bpnet_control_annotation=';'.join(sorted({acc(f['dataset']) for f in trained})), control_match='model',
                   control_plus_bigwig_accession=';'.join(f['accession'] for f in plus),
                   control_minus_bigwig_accession=';'.join(f['accession'] for f in minus))
        return exp, out, files
    # fallback (not needed for any model so far): match on the ChIP experiment's possible_controls
    ctl_exps = frozenset(acc(x['@id'] if isinstance(x, dict) else x) for x in expd.get('possible_controls', []))
    pool = ctl_by_inputs if status == 'released' else arch_ctl_by_inputs  # archived controls only for archived models
    ctl, match = sorted(pool.get(ctl_exps, []), key=lambda c: c['status'] != 'released'), 'exact'
    if not ctl:  # best overlap on control-experiment set
        scored = sorted(((len(k & ctl_exps), v) for k, v in pool.items() if k & ctl_exps),
                        key=lambda x: -x[0])
        ctl, match = (sorted(scored[0][1], key=lambda c: c['status'] != 'released'), 'partial') if scored else ([], 'none')
    if ctl:
        cf = dataset_files(ctl[0]['accession'], ctl[0]['status'])
        plus = [f for f in cf if 'plus' in f['output_type'] and f['file_format'] == 'bigWig']
        minus = [f for f in cf if 'minus' in f['output_type'] and f['file_format'] == 'bigWig']
        files += [(f, Path(cell) / 'Control') for f in plus + minus]
        out.update(bpnet_control_annotation=ctl[0]['accession'], control_match=match,
                   control_plus_bigwig_accession=';'.join(f['accession'] for f in plus),
                   control_minus_bigwig_accession=';'.join(f['accession'] for f in minus))
    return exp, out, files


with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(resolve, df.itertuples()))

for r in df.itertuples():  # peaks from the metadata itself
    add(get(f'/files/{r.peak_file_accession}/'), BASE / r.cell_line / r.tf_name)
new = {}
for exp, out, files in results:
    new[exp] = out
    for f, d in files:
        add(f, BASE / d)
    print(exp, out['bpnet_model_annotation'] or 'NO MODEL', out['control_match'], flush=True)

add_cols = pd.DataFrame.from_dict(new, orient='index')
df = df.join(add_cols, on='experiment_accession')
df.to_csv(META.with_suffix(".csv.tmp"))
os.replace(META.with_suffix(".csv.tmp"), META)  # rename: works even if the csv is held open elsewhere
(BASE / 'manifest.json').write_text(json.dumps(manifest, indent=1))
perf_manifest = {a: m for a, m in manifest.items() if m['output_type'] == 'model performance metrics'}
(BASE / 'manifest_performance.json').write_text(json.dumps(perf_manifest, indent=1))
hits_manifest = {a: m for a, m in manifest.items() if m['output_type'] == 'sequence motifs instances'}
(BASE / 'manifest_motif_instances.json').write_text(json.dumps(hits_manifest, indent=1))
regions_manifest = {a: m for a, m in manifest.items() if m['output_type'] == 'training and test regions'}
(BASE / 'manifest_training_regions.json').write_text(json.dumps(regions_manifest, indent=1))
ctl_manifest = {a: m for a, m in manifest.items() if m['output_type'].startswith('observed control profile')}
(BASE / 'manifest_controls.json').write_text(json.dumps(ctl_manifest, indent=1))
ranked_manifest = {a: m for a, m in manifest.items() if m['output_type'] == 'IDR ranked peaks'}
(BASE / 'manifest_idr_ranked.json').write_text(json.dumps(ranked_manifest, indent=1))

tot = pd.DataFrame.from_dict(manifest, orient='index')
tot['kind'] = tot.output_type
print(tot.groupby('kind')['size'].agg(['count', 'sum']).assign(GB=lambda x: x['sum'] / 1e9))
print('TOTAL GB', tot['size'].sum() / 1e9)
print(df.bpnet_model_status.replace('', 'none').value_counts().to_dict(), '| control files', len(ctl_manifest))
print('no model:', (df.bpnet_model_annotation == '').sum(), '| multi-model:',
      df.bpnet_model_annotation.str.contains(';').sum(), '| control match:', df.control_match.value_counts().to_dict())
print('DONE', flush=True)
