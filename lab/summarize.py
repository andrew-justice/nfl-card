#!/usr/bin/env python3
"""Summarize GP runs: what the search believed (training, out-of-fold) versus what validation seasons say."""
import json, sys, glob, os
rows = []
for p in sorted(sys.argv[1:] or glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', '*.json'))):
    if p.endswith('smoke.json'): continue
    R = json.load(open(p))
    bt, bv = R['best_by_train'], R['best_by_val']
    top = [r['val']['acc'] for r in R['top_val'][:10]]
    print(f"{os.path.basename(p):28s} {'NULL' if R['null'] else '    '} evals {R['evals']:6d} | best-by-train: OOF acc {bt['oof_acc']:.4f} "
          f"val acc {bt['val']['acc']:.4f} val ll {bt['val']['logloss']:.4f} | best-by-val: val acc {bv['val']['acc']:.4f} "
          f"top10% {bv['val']['acc_top10']:.3f} val ll {bv['val']['logloss']:.4f} (base {R['base_val']:.4f}) size {bv['size']}")
