"""TF-MoDISco motif matrices ("sequence motifs" file of the BPNet-model annotation) for every model in PAPER/ and
ALL_BPNET/, untrimmed and trimmed. Same annotation and status as the model already chosen in each metadata table
(../big_subset_meta.csv -> PAPER/{cell}/{TF}/, ALL_BPNET/all_bpnet_meta.csv -> ALL_BPNET/{dir}/).

Per model, into {dest}/sequence_motifs__{ACC}/ (the .h5 files and the archive itself are not kept):
  {head}/seq_motifs.{head}.fold_mean.{ENCSR}.meme/*.{PFM,CWM,CWM-PFM,hCWM,hCWM-PFM}.meme   untrimmed, w=50 (as ENCODE)
  {head}/seq_motifs.{head}.fold_mean.{ENCSR}.meme_trimmed/*.{...}.trimmed.meme           trimmed
  logs.seq_motifs.fold_mean.{ENCSR}/, README.txt
Trimming is Fi-NeMo's (finemo.data_io.trim_motif, cwm_trim_threshold 0.3 as in ENCODE's hit calling): on the h5's
contrib_scores, keep first..last position with sum|cwm| >= 0.3 * max. The same span is applied to all five variants.
Checked against ENCODE's Fi-NeMo hits (lambda 0.7) where present: trimmed offset = start - start_untrimmed (+ strand).

Usage: download_motif_matrices.py [LIMIT]   (LIMIT = first n models per set, for a smoke test). md5-checked, resumable."""
import hashlib, io, json, re, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
import h5py
import numpy as np
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths

BASE = paths.ROOT
ENC = 'https://www.encodeproject.org'
CACHE = BASE / 'ALL_BPNET' / 'metadata'
TMP = BASE / 'logs_motif_matrices' / 'tmp'
TMP.mkdir(parents=True, exist_ok=True)
TRIM = 0.3


def get(path):  # as in ALL_BPNET/build_motif_manifest.py
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


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def trim_motif(cwm):  # finemo.data_io.trim_motif; cwm is (4, W)
    score = np.abs(cwm).sum(axis=0)
    keep = np.nonzero(score >= score.max() * TRIM)[0]
    return int(keep.min()), int(keep.max()) + 1


def trim_spans(h5_bytes):
    """{meme motif name: (start, end)} from the modisco h5 (names as in the .meme / Fi-NeMo hits)."""
    spans = {}
    with h5py.File(io.BytesIO(h5_bytes), 'r') as h:
        for grp in ['pos_patterns', 'neg_patterns']:
            for pat in h.get(grp, {}):
                spans[f'{grp}.{pat}'] = trim_motif(h[grp][pat]['contrib_scores'][:].T)
    return spans


def span(spans, name):
    """Look up a motif's trim span. ENCODE's names differ between files: Fi-NeMo hits write the biosample with spaces
    ('CD14-positive monocyte') where the h5/.meme use hyphens, and some .meme files drop the pos_/neg_patterns prefix.
    Those prefix-less .meme files hold only the positive patterns (neg_patterns left out; each matrix equals its
    pos_patterns contrib_scores, r = 1.0), so a bare name is a pos_patterns one."""
    name = name.replace(' ', '-')
    return spans[name if '.' in name else 'pos_patterns.' + name]


def write_trimmed(meme_text, spans):
    """Rewrite a MEME file keeping rows start..end of each motif (spans keyed by the MOTIF name)."""
    out, lines, i = [], meme_text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('MOTIF'):
            s, e = span(spans, line[len('MOTIF'):].strip())
            hdr = lines[i + 1]
            w = int(re.search(r'w= (\d+)', hdr).group(1))
            out += [line, re.sub(r'w= \d+', f'w= {e - s}', hdr)] + lines[i + 2 + s:i + 2 + e]
            i += 2 + w
        else:
            out.append(line)
            i += 1
    return '\n'.join(out) + '\n'


def check_hits(d, spans_by_head):
    """Compare trim spans with ENCODE Fi-NeMo hit offsets (lambda 0.7). Returns (n motifs checked, n mismatched)."""
    n = bad = 0
    for head, spans in spans_by_head.items():
        for tsv in d.glob(f'sequence_motifs_instances__*/{head}/seq_motifs_instances.{head}.lambda_0p7/*.tsv'):
            hits = pd.read_csv(tsv, sep='\t', usecols=['start', 'end', 'start_untrimmed', 'end_untrimmed', 'motif_name', 'strand'])
            hits = hits[hits.strand == '+']
            off = hits.assign(s=hits.start - hits.start_untrimmed, e=hits.end - hits.start_untrimmed).groupby('motif_name')[['s', 'e']]
            for name, g in off:
                n += 1
                s, e = span(spans, name)
                if not ((g.s == s).all() and (g.e == e).all()):
                    bad += 1
                    print('  TRIM MISMATCH', d, head, name, (s, e), sorted(set(zip(g.s, g.e)))[:3], flush=True)
    return n, bad


def fetch(f, d):
    acc = f['accession']
    xdir = d / f'sequence_motifs__{acc}'
    if (xdir / '.done').exists():
        return acc, 'already done', 0, 0
    tmp = TMP / f'{acc}.tar.gz'
    for attempt in range(3):
        if not (tmp.exists() and md5(tmp) == f['md5sum']):
            subprocess.run(['curl', '-sSL', '--retry', '5', '-C', '-', '-o', str(tmp), ENC + f['href']], check=False)
        if tmp.exists() and md5(tmp) == f['md5sum']:
            break
        tmp.unlink(missing_ok=True)
    else:
        return acc, 'FAILED md5/download', 0, 0
    spans_by_head, memes = {}, []
    with tarfile.open(tmp, 'r:*') as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            p = Path(m.name.lstrip('./'))
            if p.suffix == '.h5':  # seq_motifs.{head}.fold_mean.{ENCSR}.h5 -> trim spans only, not written
                head = p.name.split('.')[1]
                spans_by_head[head] = trim_spans(t.extractfile(m).read())
                continue
            out = xdir / p
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(t.extractfile(m).read())
            if p.suffix == '.meme':
                memes.append(out)
    for mf in memes:  # .../{head}/seq_motifs.{head}.fold_mean.{ENCSR}.meme/{...}.CWM.meme
        head = mf.parent.parent.name
        tdir = mf.parent.with_name(mf.parent.name + '_trimmed')
        tdir.mkdir(exist_ok=True)
        (tdir / mf.name.replace('.meme', '.trimmed.meme')).write_text(write_trimmed(mf.read_text(), spans_by_head[head]))
    n, bad = check_hits(d, spans_by_head)
    (xdir / '.done').touch()
    tmp.unlink()
    return acc, f'done memes={len(memes)} motifs={sum(map(len, spans_by_head.values()))}', n, bad


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    files = {}
    for f in get('/search/?type=File&output_type=sequence+motifs&status=released&status=archived&frame=object&limit=all')['@graph']:
        files.setdefault(f['dataset'].strip('/').split('/')[-1], []).append(f)

    paper = pd.read_csv(paths.META_PAPER, keep_default_na=False)
    paper = paper[paper.bpnet_model_annotation != ''].assign(dest=lambda x: 'PAPER/' + x.cell_line + '/' + x.tf_name)
    allb = pd.read_csv(paths.META_ALL, keep_default_na=False).assign(dest=lambda x: 'ALL_BPNET/' + x.dir)
    jobs, missing, multi = [], [], []
    for name, meta in [('PAPER', paper), ('ALL_BPNET', allb)]:
        for r in meta.head(limit).itertuples() if limit else meta.itertuples():
            mf = sorted((f for a in r.bpnet_model_annotation.split(';') for f in files.get(a, [])
                         if f['status'] == r.bpnet_model_status), key=lambda f: f['accession'])
            if not mf:
                missing.append(r.dest)
            elif len(mf) > 1:
                multi.append((r.dest, [f['accession'] for f in mf]))
            jobs += [(f, BASE / r.dest) for f in mf]
    print('models PAPER', len(paper), 'ALL_BPNET', len(allb), '| files to fetch', len(jobs),
          f'{sum(f.get("file_size", 0) for f, _ in jobs) / 1e9:.1f} GB', flush=True)
    print('NO sequence motifs file:', len(missing), missing, flush=True)
    print('several files:', multi, flush=True)

    failed, n_chk, n_bad = [], 0, 0
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(fetch, f, d): f['accession'] for f, d in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                acc, status, n, bad = fu.result()
            except Exception as e:
                acc, status, n, bad = futs[fu], f'FAILED {e!r}', 0, 0
            print(i, acc, status, f'hits-checked={n} mismatched={bad}', flush=True)
            n_chk, n_bad = n_chk + n, n_bad + bad
            if status.startswith('FAILED'):
                failed.append(acc)
    print(f'TRIM CHECK motifs={n_chk} mismatched={n_bad}', flush=True)
    print('FAILED', failed, flush=True)
    print('DONE', flush=True)
