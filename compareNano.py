from __future__ import print_function

import argparse
import uproot
import numpy as np
import awkward as ak

parser = argparse.ArgumentParser('Compare two nanoAOD files')
parser.add_argument('file1')
parser.add_argument('file2')
args = parser.parse_args()

t1 = uproot.open(args.file1)['Events']
t2 = uproot.open(args.file2)['Events']

br1 = set(t1.keys())
br2 = set(t2.keys())

print(f'=== Branches in {args.file1} only ===\n - ' + ('\n - '.join(sorted(br1 - br2))))
print(f'=== Branches in {args.file2} only ===\n - ' + ('\n - '.join(sorted(br2 - br1))))

branches = sorted(br1 & br2)

print('====== Diffs ======')
for k in branches:
    a1 = t1[k].array()
    a2 = t2[k].array()
    if a1.ndim == 1:
        a1 = ak.to_numpy(a1)
        a2 = ak.to_numpy(a2)
    else:
        a1_offsets, a1_content = a1.layout.offsets.data, a1.layout.content.data
        a2_offsets, a2_content = a2.layout.offsets.data, a2.layout.content.data
    if isinstance(a1, np.ndarray):
        same = np.all(a1 == a2)
    else:
        same = np.all(a1_offsets == a2_offsets) and np.all(a1_content == a2_content)
    if not same:
        if isinstance(a1, np.ndarray):
            close = np.allclose(a1, a2, rtol=1e-3, atol=1e-3, equal_nan=True)
        else:
            close = np.all(a1_offsets == a2_offsets) and np.allclose(
                a1_content, a2_content, rtol=1e-3, atol=1e-3, equal_nan=True)
        print(k, '(close)' if close else '')
        print(' ... a1=%s\n ... a2=%s' % (a1[:5], a2[:5]))
