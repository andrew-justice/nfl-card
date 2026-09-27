#!/usr/bin/env python3
"""
gafs.py -- genetic algorithm over feature subsets for a ridge-logistic model.

Genome: one bit per feature plus a regularization gene.  Fitness: out-of-fold log loss across
interleaved training-season folds (2002-2015) plus a small per-feature penalty, so the GA is pushed
toward compact models that hold up on seasons they were not fitted on.  Validation (2016-2020)
only ranks the hall of fame at the end.

usage: python3 gafs.py --task su_ind --seed 1 --gens 60 --pop 120 --out runs/gafs-su_ind-s1.json [--null]
"""
import numpy as np, json, random, time, argparse, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp

CS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

def irls(X, y, ridge, iters=12):
    """ridge logistic, intercept unpenalized; X already standardized with a leading column of ones"""
    b = np.zeros(X.shape[1]); R = ridge * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        z = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-z)); W = p * (1 - p) + 1e-9
        H = (X.T * W) @ X + R; g = X.T @ (y - p) - R @ b
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-6: break
    return b

class Fit:
    def __init__(self, P, pen):
        self.P, self.pen = P, pen
        tr = P['tr']
        self.Z, self.y, self.fold = P['Z'][tr], P['y'][tr], P['fold'][tr]
        self.cache = {}

    def score(self, mask, ci):
        key = (mask.tobytes(), ci)
        if key in self.cache: return self.cache[key]
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            self.cache[key] = (9.0, 0.0); return self.cache[key]
        X = np.column_stack([np.ones(len(self.y)), self.Z[:, idx]])
        oof = np.zeros(len(self.y)); ridge = 1.0 / CS[ci]
        for f in range(4):
            m = self.fold == f
            b = irls(X[~m], self.y[~m], ridge); oof[m] = X[m] @ b
        ll = gp.logloss(oof, self.y); acc = float(np.mean((oof > 0) == (self.y > 0.5)))
        self.cache[key] = (ll + self.pen * len(idx), acc)
        return self.cache[key]

def full(P, mask, ci, rows):
    idx = np.flatnonzero(mask)
    X = np.column_stack([np.ones(len(P['y'])), P['Z'][:, idx]])
    b = irls(X[rows], P['y'][rows], 1.0 / CS[ci])
    return b, X @ b

def run(task, seed, gens, pop, null=False, pen=2e-5):
    D = gp.load(); P = gp.prepare(D, task, null_seed=(2000 + seed) if null else None)
    rng = np.random.default_rng(seed); R = random.Random(seed)
    F = len(P['names']); S = Fit(P, pen)
    def rand():
        return (rng.random(F) < R.uniform(0.05, 0.3)), R.randrange(len(CS))
    popn = [rand() for _ in range(pop)]
    hof = {}; t0 = time.time(); log = []
    for gen in range(gens):
        scored = []
        for m, c in popn:
            fit, acc = S.score(m, c); scored.append((fit, acc, m, c))
            hof[(m.tobytes(), c)] = (fit, acc, m, c)
        scored.sort(key=lambda x: x[0])
        nxt = [(s[2].copy(), s[3]) for s in scored[:4]]
        def tour():
            c = R.sample(scored, 4); return min(c, key=lambda x: x[0])
        while len(nxt) < pop:
            a, b = tour(), tour()
            if R.random() < 0.7:
                u = rng.random(F) < 0.5; child = np.where(u, a[2], b[2]); ci = a[3] if R.random() < 0.5 else b[3]
            else:
                child = a[2].copy(); ci = a[3]
            flip = rng.random(F) < (2.0 / F); child = child ^ flip
            if R.random() < 0.15: ci = min(len(CS) - 1, max(0, ci + R.choice((-1, 1))))
            nxt.append((child, ci))
        popn = nxt
        if (gen + 1) % 10 == 0 or gen == 0:
            b = scored[0]
            log.append(dict(gen=gen + 1, fit=round(b[0], 5), oof_acc=round(b[1], 4), nfeat=int(b[2].sum()), C=CS[b[3]]))
            print(f"[gafs {task}{' NULL' if null else ''} s{seed}] gen {gen + 1}: fit {b[0]:.5f} OOF acc {b[1]:.4f} features {int(b[2].sum())} C {CS[b[3]]} ({time.time() - t0:.0f}s)", flush=True)
    top = sorted(hof.values(), key=lambda x: x[0])[:150]
    res = []
    for fit, acc, m, c in top:
        b, z = full(P, m, c, P['tr'])
        res.append(dict(fit=round(fit, 6), oof_acc=round(acc, 4), nfeat=int(m.sum()), C=CS[c],
                        features=[P['names'][j] for j in np.flatnonzero(m)],
                        train=gp.metrics(z, P['y'], P['tr']), val=gp.metrics(z, P['y'], P['va'])))
    by_val = sorted(res, key=lambda r: (r['val']['logloss'], r['nfeat']))
    return dict(task=task, seed=seed, null=null, gens=gens, pop=pop, pen=pen, log=log, evals=len(S.cache),
                best_by_train=res[0], best_by_val=by_val[0], top_val=by_val[:20])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gens', type=int, default=60); ap.add_argument('--pop', type=int, default=120)
    ap.add_argument('--pen', type=float, default=2e-5); ap.add_argument('--null', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = run(a.task, a.seed, a.gens, a.pop, null=a.null, pen=a.pen)
    json.dump(out, open(a.out, 'w'))
    bt, bv = out['best_by_train'], out['best_by_val']
    print(f"DONE gafs {a.task}{' NULL' if a.null else ''} s{a.seed}: best-by-train OOF acc {bt['oof_acc']} val acc {bt['val']['acc']} "
          f"({bt['nfeat']} features) | best-by-val val acc {bv['val']['acc']} val ll {bv['val']['logloss']} ({bv['nfeat']} features, C {bv['C']})")
    print('  features:', ', '.join(bv['features']))
