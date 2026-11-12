"""Collect {BASE}/{dir}/testset_auc_{tag}.csv into {BASE}/testset_auc_all.csv (per fold) and
{BASE}/testset_auc_summary.csv (mean over folds, with ENCODE's with-control values next to the no-control ones, plus the
experiment and the model's QC status); if present, also testset_auc_mix_{tag}.csv -> testset_auc_mix_all.csv, merged
into the summary, and testset_auc_shuf_{tag}.csv -> testset_auc_shuf_all.csv (shuffled-sequence columns). Model set: env BPNET_SET (paper by default, or all; see bpnet_sets.py)."""
import os, re
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

BASE, meta = bpnet_sets.load()
if 'bpnet_model_qc_status' not in meta:  # PAPER metadata: parse it from the annotation's QC note
    meta['bpnet_model_qc_status'] = meta.bpnet_model_qc.map(lambda q: ';'.join(sorted(set(re.findall(r'status:(\w+)', q)))))


def collect(kind):
    fs = [f'{BASE}/{r.dir}/testset_auc{kind}_{r.tag}.csv' for r in meta.itertuples()]
    fs = [f for f in fs if os.path.exists(f)]
    print(f'testset_auc{kind}: {len(fs)} of {len(meta)} experiments', flush=True)
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


df = collect('')
df.to_csv(f'{BASE}/testset_auc_all.csv', index=False)
cols = ['encode_auroc', 'auroc_noctl', 'encode_auprc', 'auprc_noctl', 'auprc_baseline', 'baseline_absdiff']
summ = df.groupby(['tf_cells', 'model_status'])[cols].mean().reset_index()
summ = summ.merge(meta[['tf_cells', 'experiment_accession', 'bpnet_model_qc_status']], on='tf_cells', how='left')
summ.to_csv(f'{BASE}/testset_auc_summary.csv', index=False)
print('TFs', len(summ), '| folds', len(df), '| max baseline_absdiff', df.baseline_absdiff.max().round(4),
      '| folds with absdiff > 0.005:', (df.baseline_absdiff > 0.005).sum())
print(summ.groupby('model_status')[cols].median().round(3).to_string())

# with control, sequences mixed between regions (testset_auc_mix.py)
mix = collect('_mix')
if mix is not None:
    mix.to_csv(f'{BASE}/testset_auc_mix_all.csv', index=False)
    mcols = ['auroc_ctl', 'auroc_mix', 'auprc_ctl', 'auprc_mix', 'ctl_absdiff']
    msumm = mix.groupby(['tf_cells', 'model_status'])[mcols].mean().reset_index()
    summ = summ.merge(msumm, on=['tf_cells', 'model_status'], how='left')
    # fraction of the model's above-chance AUROC that survives mixing: ~1 control-driven, ~0 sequence-driven
    summ['ctl_share'] = (summ.auroc_mix - 0.5) / (summ.auroc_ctl - 0.5)
    summ.to_csv(f'{BASE}/testset_auc_summary.csv', index=False)
    print('mix TFs', len(msumm), '| folds', len(mix), '| max ctl_absdiff (auroc_ctl vs ENCODE)', mix.ctl_absdiff.max().round(4),
          '| folds > 0.005:', (mix.ctl_absdiff > 0.005).sum())
    print(summ.groupby('model_status')[['encode_auroc', 'auroc_ctl', 'auroc_mix', 'auroc_noctl',
                                        'encode_auprc', 'auprc_ctl', 'auprc_mix', 'auprc_noctl', 'ctl_share']].median().round(3).to_string())
    print(summ.groupby('bpnet_model_qc_status')[['auroc_ctl', 'auroc_mix', 'auroc_noctl', 'ctl_share']]
          .agg(['median', 'size']).round(3).to_string())

# dinucleotide-shuffled sequences, ± control (testset_auc_shuf.py)
shuf = collect('_shuf')
if shuf is not None:
    shuf.to_csv(f'{BASE}/testset_auc_shuf_all.csv', index=False)
    scols = [f'{m}_{s}{c}' for c in ['ctl', 'noctl'] for s in ['shufpk_', 'shufall_'] for m in ['auroc', 'auprc']]
    ssumm = shuf.groupby(['tf_cells', 'model_status'])[scols].mean().reset_index()
    summ = summ.merge(ssumm, on=['tf_cells', 'model_status'], how='left')
    summ.to_csv(f'{BASE}/testset_auc_summary.csv', index=False)
    print('shuf TFs', len(ssumm), '| folds', len(shuf))
    print(summ.groupby('bpnet_model_qc_status')[['auroc_ctl', 'auroc_shufpk_ctl', 'auroc_shufall_ctl', 'auroc_noctl',
                                                 'auroc_shufpk_noctl', 'auroc_shufall_noctl']].median().round(3).to_string())
