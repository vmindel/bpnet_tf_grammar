"""Metadata + download manifest for every ENCODE BPNet model NOT already in big_subset_meta.csv (the PAPER set,
kept separate). Only the files needed for predictions with/without control and the test-set AUROC/AUPRC:
  models tar, model performance metrics tar, training and test regions tar  (from the BPNet-model annotation)
  conservative IDR thresholded peaks (GRCh38 bed)                            (peaks for predict_paper-style runs)
  pooled-pseudoreplicated IDR ranked peaks (GRCh38 bed)                      (ambiguous-nonpeak filter)
  observed control profile bigWigs (plus/minus) the models file was derived from.
One row per experiment: its released BPNet model, else its archived (QC-failed) one, as in PAPER.
Uses bulk ENCODE searches (frame=object), cached in metadata/. Nothing is downloaded.
Kept: experiments with conservative peaks + pooled-pseudoreplicate ranked peaks (replicated) and a control
(unreplicated / control-less ones are listed in excluded.csv). Layout: {cell}/{TF}_{experiment}/ and {cell}/Control/.
Files already downloaded for PAPER are symlinked from there, not downloaded again.
Writes all_bpnet_meta.csv, excluded.csv and manifest.json (to download: accession -> href, md5, size, output_type, dest).
Download: download_all.py manifest.json"""
import json, re, time
import download_all
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths

BASE = paths.ALL_BPNET
PAPER_META = paths.META_PAPER
PAPER_MANIFEST = paths.PAPER / 'manifest.json'
CACHE = BASE / 'metadata'
CACHE.mkdir(exist_ok=True)
ENC = 'https://www.encodeproject.org'


def get(path):
    dest = CACHE / (re.sub(r'[^A-Za-z0-9_.-]', '_', path.strip('/'))[:200] + '.json')
    if dest.exists():
        return json.loads(dest.read_text())
    for attempt in range(4):
        try:
            with urlopen(Request(ENC + path, headers={'Accept': 'application/json'}), timeout=600) as r:
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


models = search('type=Annotation&annotation_type=BPNet-model')
exps = {e['accession']: e for e in search('type=Experiment&assay_term_name=ChIP-seq'
                                          '&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens')}
targets = {t['@id']: t['label'] for t in search('type=Target')}
biosamples = {b['@id']: b['term_name'] for b in search('type=BiosampleType')}
by_type = {}
for ot in ['models', 'model performance metrics', 'training and test regions',
           'observed control profile (plus strand)', 'observed control profile (minus strand)']:
    by_type[ot] = search('type=File&output_type=' + ot.replace(' ', '+').replace('(', '%28').replace(')', '%29'))
    print(ot, len(by_type[ot]), flush=True)
peak_files = search('type=File&output_type=conservative+IDR+thresholded+peaks&assembly=GRCh38&file_format=bed')
opt_peak_files = search('type=File&output_type=IDR+thresholded+peaks&assembly=GRCh38&file_format=bed')
ranked_files = search('type=File&output_type=IDR+ranked+peaks&assembly=GRCh38&file_format=bed')
print('peaks', len(peak_files), '| ranked', len(ranked_files), flush=True)

ann_files = {}
for ot in ['models', 'model performance metrics', 'training and test regions']:
    for f in by_type[ot]:
        ann_files.setdefault(acc(f['dataset']), []).append(f)
ctl_file = {f['@id']: f for ot in by_type if ot.startswith('observed') for f in by_type[ot]}
# unreplicated experiments (mostly tissues) have no conservative peaks / pooled-pseudoreplicate ranked peaks;
# fall back to their IDR thresholded peaks / pseudoreplicate IDR ranked peaks (peak_source/ranked_source say which)
exp_peaks, exp_opt_peaks, exp_ranked, exp_pr_ranked = {}, {}, {}, {}
for files, pool in [(peak_files, exp_peaks), (opt_peak_files, exp_opt_peaks)]:
    for f in files:
        if f['status'] == 'released':
            pool.setdefault(acc(f['dataset']), []).append(f)
for f in ranked_files:
    step = f.get('analysis_step_version') or ''
    if f['status'] == 'released' and 'pseudoreplicated' in step:
        (exp_ranked if 'pooled-pseudoreplicated' in step else exp_pr_ranked).setdefault(acc(f['dataset']), []).append(f)

paper_exps = set(pd.read_csv(PAPER_META, index_col=0).experiment_accession)
# already downloaded for PAPER (any manifest, or a control bigWig on disk): not downloaded again, reused from there
paper_files = {a for p in PAPER_MANIFEST.parent.glob('manifest*.json') for a in json.loads(p.read_text())}
paper_files |= {p.name.split('__')[-1].split('.')[0] for p in PAPER_MANIFEST.parent.glob('*/Control/*.bigWig')}

by_exp = {}
for m in models:
    for x in m['experimental_input']:
        by_exp.setdefault(acc(x), []).append(m)

rows, manifest, skipped = [], {}, {'in_paper': 0, 'not_human_chip': 0}


def add(f, dest):
    manifest[f['accession']] = {'href': f['href'], 'md5': f['md5sum'], 'size': f.get('file_size', 0),
                                'output_type': f['output_type'], 'dest': dest, 'in_paper': f['accession'] in paper_files}


for exp, ms in sorted(by_exp.items()):
    if exp in paper_exps:
        skipped['in_paper'] += 1
        continue
    if exp not in exps:
        skipped['not_human_chip'] += 1
        continue
    e = exps[exp]
    status = 'released' if any(m['status'] == 'released' for m in ms) else 'archived'
    ms = [m for m in ms if m['status'] == status]
    cell, tf = biosamples.get(e['biosample_ontology'], acc(e['biosample_ontology'])), targets.get(e.get('target'), 'NA')
    cdir = re.sub(r'[^A-Za-z0-9-]+', '_', cell).strip('_')
    qc = ';'.join(m.get('description', '').split(f'({exp})', 1)[-1] for m in ms)
    mf = [f for m in ms for f in ann_files.get(m['accession'], []) if f['status'] == status]
    pick = lambda ot: sorted((f for f in mf if f['output_type'] == ot), key=lambda f: f['accession'])
    mod, perf, reg = pick('models'), pick('model performance metrics'), pick('training and test regions')
    ctl = [ctl_file[p] for f in mod for p in f.get('derived_from', []) if p in ctl_file]
    plus = [f for f in ctl if 'plus' in f['output_type']]
    minus = [f for f in ctl if 'minus' in f['output_type']]
    peak_source, peaks = next(((k, v) for k, v in [('conservative', exp_peaks.get(exp)),
                                                    ('optimal', exp_opt_peaks.get(exp))] if v), ('none', []))
    peaks = sorted(peaks, key=lambda f: (not f.get('preferred_default'), f['accession']))[:1]
    ranked_source, ranked = next(((k, v) for k, v in [('pooled_pseudorep', exp_ranked.get(exp)),
                                                      ('pseudorep', exp_pr_ranked.get(exp))] if v), ('none', []))
    rows.append(dict(experiment_accession=exp, experiment_status=e['status'], tf_name=tf, biosample_ontology=cell,
                     cell_line=cdir, dir=f'{cdir}/{tf}_{exp}', bpnet_model_status=status,
                     bpnet_model_qc_status=';'.join(sorted(set(re.findall(r'status:(\w+)', qc)))), bpnet_model_qc=qc,
                     bpnet_model_annotation=';'.join(m['accession'] for m in ms),
                     bpnet_models_accession=';'.join(f['accession'] for f in mod),
                     model_performance_accession=';'.join(f['accession'] for f in perf),
                     training_regions_accession=';'.join(f['accession'] for f in reg),
                     peak_file_accession=';'.join(f['accession'] for f in peaks), peak_source=peak_source,
                     idr_ranked_peaks_accession=';'.join(sorted(f['accession'] for f in ranked)), ranked_source=ranked_source,
                     control_plus_bigwig_accession=';'.join(f['accession'] for f in plus),
                     control_minus_bigwig_accession=';'.join(f['accession'] for f in minus)))
    if peak_source != 'conservative' or not plus or not minus:
        continue
    for f in mod + perf + reg + peaks + ranked:
        add(f, f'{cdir}/{tf}_{exp}')
    for f in plus + minus:
        add(f, f'{cdir}/Control')

meta = pd.DataFrame(rows)
keep = (meta.peak_source == 'conservative') & (meta.control_plus_bigwig_accession != '') & (meta.control_minus_bigwig_accession != '')
meta[~keep].assign(reason=lambda x: x.peak_source.where(x.peak_source != 'conservative', 'no_control')
                   .replace('optimal', 'unreplicated')).to_csv(paths.METADATA / 'excluded.csv', index=False)
print('excluded', (~keep).sum(), '(see excluded.csv)')
meta = meta[keep]
meta.to_csv(paths.META_ALL, index=False)

# files already in PAPER: symlink (controls live in PAPER/{cell}/Control/)
paper_ctl = {p.name.split('__')[-1].split('.')[0]: p for p in PAPER_MANIFEST.parent.glob('*/Control/*.bigWig')}
for a, m in manifest.items():
    if m['in_paper']:
        assert a in paper_ctl, f'{a} in PAPER but not a control bigWig on disk'
        stem, ext = download_all.local_name(a, m)
        link = BASE / m['dest'] / f'{stem}.{ext}'
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(paper_ctl[a])
(BASE / 'manifest.json').write_text(json.dumps({a: m for a, m in manifest.items() if not m['in_paper']}, indent=1))

print('model annotations', len(models), '| experiments with a model', len(by_exp), '| skipped', skipped)
print('new experiments', len(meta), meta.bpnet_model_status.value_counts().to_dict())
for c in ['bpnet_models_accession', 'model_performance_accession', 'training_regions_accession', 'peak_file_accession',
          'idr_ranked_peaks_accession', 'control_plus_bigwig_accession', 'control_minus_bigwig_accession']:
    print(f'  missing {c}: {(meta[c] == "").sum()} | multiple: {meta[c].str.contains(";").sum()}')
print('QC:', meta.groupby(['bpnet_model_status', 'bpnet_model_qc_status']).size().to_dict())
print('top cell lines:', meta.biosample_ontology.value_counts().head(15).to_dict())
tot = pd.DataFrame.from_dict(manifest, orient='index')
g = tot.groupby('output_type').agg(files=('size', 'size'), GB=('size', lambda s: s.sum() / 1e9),
                                   files_in_paper=('in_paper', 'sum'),
                                   GB_new=('size', lambda s: s[~tot.loc[s.index, 'in_paper']].sum() / 1e9))
print(g.round(2).to_string())
print('TOTAL GB', round(tot['size'].sum() / 1e9, 1), '| not already in PAPER', round(tot['size'][~tot.in_paper].sum() / 1e9, 1))
print('DONE', flush=True)
