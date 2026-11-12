"""Download everything in manifest.json (built by build_all_table.py) into ALL_BPNET/{cell}/{TF}_{experiment}/ and
ALL_BPNET/{cell}/Control/ (copy of PAPER/download_paper.py). Each accession is listed once, so shared controls are fetched once.
md5-verified, resumable; tars are extracted into {output_type}__{accession}/ and the archive deleted."""
import hashlib, json, re, subprocess, sys, tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths

BASE = paths.ALL_BPNET
ENC = 'https://www.encodeproject.org'


def local_name(acc, m):
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


def fetch(acc):
    m = manifest[acc]
    d = BASE / m['dest']
    d.mkdir(parents=True, exist_ok=True)
    stem, ext = local_name(acc, m)
    is_tar = ext.startswith('tar')
    out, xdir = d / f'{stem}.{ext}', d / stem
    if is_tar and xdir.is_dir() and any(xdir.iterdir()) and not out.exists():
        return acc, 'already extracted'
    if not (out.exists() and md5(out) == m['md5']):
        for attempt in range(3):
            subprocess.run(['curl', '-sSL', '--retry', '5', '-C', '-', '-o', str(out), ENC + m['href']], check=False)
            if out.exists() and md5(out) == m['md5']:
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


if __name__ == '__main__':
    manifest = json.loads((BASE / (sys.argv[1] if len(sys.argv) > 1 else 'manifest.json')).read_text())
    print('files to fetch', len(manifest), flush=True)
    failed = []
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(fetch, a) for a in manifest]
        for i, fu in enumerate(as_completed(futs), 1):
            acc, status = fu.result()
            print(i, acc, manifest[acc]['dest'], status, flush=True)
            if status.startswith('FAILED'):
                failed.append(acc)
    print('FAILED', failed, flush=True)
    print('DONE', flush=True)
