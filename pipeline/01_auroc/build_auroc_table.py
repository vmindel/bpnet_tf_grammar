"""data/auroc_summary.parquet (Fig. 1A): one row per BPNet model (PAPER 440 + ALL_BPNET 1,842), the per-experiment
test-set AUROC/AUPRC summaries written by collect_testset_auc.py ({set}/testset_auc_summary.csv), stacked.
  auroc_ctl    with the model's own control track (testset_auc_mix.py; reproduces ENCODE's reported value,
               see ctl_absdiff)
  auroc_noctl  with an all-zero control track (testset_auc.py)
Mean over the 5 test folds. These are the same values as the auroc_* columns of the project's full model table
(bpnet_big_table.parquet), which also needs the motif-knockout runs that no figure uses. CPU job."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules'))
import paths

summ = pd.concat([pd.read_csv(paths.ROOT / d / 'testset_auc_summary.csv').assign(set=label)
                  for d, label in [('PAPER', 'PAPER'), ('ALL_BPNET', 'ALL_OTHER')]], ignore_index=True)
assert len(summ) == 440 + 1842 and summ.auroc_ctl.notna().all() and summ.auroc_noctl.notna().all(), len(summ)
summ = summ[['set'] + [c for c in summ if c != 'set']]
paths.DATA.mkdir(exist_ok=True)
summ.to_parquet(paths.DATA / 'auroc_summary.parquet', index=False)
print(summ.shape, summ.set.value_counts().to_dict())
print(summ[['auroc_ctl', 'auroc_noctl', 'auroc_mix', 'ctl_share']].median().round(3).to_dict())
