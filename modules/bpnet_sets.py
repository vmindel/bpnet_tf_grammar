"""Which model set the prediction / AUROC scripts run on, chosen by env BPNET_SET:
  paper (default)  PAPER/, metadata/big_subset_meta.csv           dir = {cell}/{TF}
  all              ALL_BPNET/, metadata/all_bpnet_meta.csv  dir = {cell}/{TF}_{experiment}
load() -> (BASE, meta): experiments that have a BPNet model, with columns dir (under BASE), tag (dir with '/' -> '_',
used in output file names) and tf_cells (unique label), sorted by dir. Controls are in BASE/{cell_line}/Control/.
select(meta, argv) -> the rows named on the command line (as dir), else the LSF array slice LSB_JOBINDEX of N_TASKS."""
import os
import pandas as pd
import paths

ROOT = str(paths.ROOT)
SET = os.environ.get('BPNET_SET', 'paper')


def load():
    if SET == 'paper':
        base = f'{ROOT}/PAPER'
        meta = pd.read_csv(paths.META_PAPER, index_col=0)
        meta = meta[meta.bpnet_model_status.notna() & (meta.bpnet_model_status != '')]
        meta['dir'] = meta.cell_line + '/' + meta.tf_name
    elif SET == 'all':
        base = f'{ROOT}/ALL_BPNET'
        meta = pd.read_csv(paths.META_ALL, keep_default_na=False)
        meta['tf_cells'] = meta.tf_name + '_' + meta.cell_line + '_' + meta.experiment_accession
    else:
        raise ValueError(f'BPNET_SET={SET!r}: expected paper or all')
    meta['tag'] = meta.dir.str.replace('/', '_')
    assert meta.dir.is_unique and meta.tf_cells.is_unique
    return base, meta.sort_values('dir')


def select(meta, argv):
    if argv:
        return meta[meta.dir.isin(argv)]
    return meta.iloc[int(os.environ['LSB_JOBINDEX']) - 1::int(os.environ.get('N_TASKS', 1))]
