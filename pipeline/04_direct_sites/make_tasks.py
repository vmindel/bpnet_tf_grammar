"""Split experiments into array tasks balanced by peak count (greedy, largest first) -> tasks_{set}{SUFFIX}.txt, one line
of dirs per task. Usage: make_tasks.py [PEAKS_PER_TASK [SUFFIX]]; SUFFIX _direct keeps only experiments with a bigBed."""
import sys, glob, heapq, pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
PEAKS_PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000   # epistasis ran ~25 peaks/s at 250k/task (~3 h); direct_sites is ~1.4x slower (FOXA1 892 vs 635 s) -> 150k ~2.5 h
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ''
BASE = {'paper': paths.PAPER, 'all': paths.ALL_BPNET}
HERE = _Path(__file__).resolve().parent
pc = pd.read_csv(HERE.parent / '03_epistasis' / 'tasks' / 'peak_counts.csv')
if SUFFIX == '_direct':
    pc = pc[[bool(glob.glob(f'{BASE[s]}/{d}/sequence_motifs_instances__*.bigBed')) for s, d in zip(pc.set, pc.dir)]]
for s, g in pc.groupby('set'):
    n = max(1, round(g.n_peaks.sum() / PEAKS_PER_TASK))
    heap = [(0, i, []) for i in range(n)]
    for r in g.sort_values('n_peaks', ascending=False).itertuples():
        tot, i, dirs = heapq.heappop(heap)
        heapq.heappush(heap, (tot + r.n_peaks, i, dirs + [r.dir]))
    heap.sort(key=lambda t: t[1])
    open(HERE / 'tasks' / f'tasks_{s}{SUFFIX}.txt', 'w').write(''.join(' '.join(d) + '\n' for _, _, d in heap))
    print(s, len(g), 'experiments', n, 'tasks, peaks/task max', max(t for t, _, _ in heap), 'min', min(t for t, _, _ in heap))
