#!/usr/bin/env python3
"""
gp.py -- multi-gene genetic programming (MGGP) search for NFL prediction formulas.

An individual is 1-4 evolved formula trees ("genes"); their outputs are combined by a fitted
logistic (or least-squares) layer.  Fitness is out-of-fold loss across interleaved training-season
folds (2002-2015), so a formula only scores well if it predicts seasons it was not fitted on.
Validation seasons (2016-2020) are used only to choose among the hall of fame at the end.
Test seasons (2021-2025) are never touched here.

usage: python3 gp.py --task su_ind --seed 1 [--gens 80 --pop 300 --islands 4] [--null] --out runs/x.json
"""
import numpy as np, csv, json, math, random, time, argparse, os

LAB = os.path.dirname(os.path.abspath(__file__))
MARKET = {'spread', 'total_line', 'ml_logit', 'ml_gap', 'juice_s', 'juice_t', 'imp_h', 'imp_a', 'model_gap', 'model_gap_t', 'has_line'}
TRAIN, VAL, TEST = (2002, 2015), (2016, 2020), (2021, 2025)

TASKS = {
    'su_ind': dict(target='y_win', exclude=MARKET, desc='straight-up winner, no market inputs'),
    'su_mkt': dict(target='y_win', exclude=set(), desc='straight-up winner, market inputs allowed'),
    'ats': dict(target='y_cover', exclude=set(), desc='home covers the closing spread'),
    'ou': dict(target='y_over', exclude=set(), desc='game goes over the closing total'),
    'margin': dict(target='y_margin', exclude=MARKET, desc='home margin in points, no market inputs', kind='reg'),
    'total': dict(target='y_total', exclude=MARKET, desc='total points, no market inputs', kind='reg'),
}
def is_reg(task): return TASKS[task].get('kind') == 'reg'

# ------------------------------------------------------------------ data
def load(path=os.path.join(LAB, 'feats.csv')):
    with open(path, newline='') as fh:
        rd = csv.reader(fh); head = next(rd); rows = list(rd)
    idx = {c: i for i, c in enumerate(head)}
    fcols = [c for c in head if c.startswith('f_')]
    X = np.array([[float(r[idx[c]]) if r[idx[c]] != '' else np.nan for c in fcols] for r in rows])
    col = lambda c: np.array([float(r[idx[c]]) if (c in idx and r[idx[c]] != '') else np.nan for r in rows])
    D = dict(X=X, names=[c[2:] for c in fcols], season=col('season').astype(int), week=col('week').astype(int),
             final=col('final').astype(int), Y={c: col(c) for c in ('y_margin', 'y_total', 'y_win', 'y_ats', 'y_cover', 'y_ou', 'y_over')},
             game_id=[r[idx['game_id']] for r in rows], home=[r[idx['home']] for r in rows], away=[r[idx['away']] for r in rows])
    return D

def prepare(D, task, null_seed=None, base=False):
    T = TASKS[task]
    y = D['Y'][T['target']].copy()
    s = D['season']
    if null_seed is not None:
        # coin-flip world: shuffle outcomes within each season, destroying any real signal but keeping base rates
        rng = np.random.default_rng(null_seed)
        for yr in np.unique(s):
            m = (s == yr) & ~np.isnan(y)
            y[m] = rng.permutation(y[m])
    feats = [j for j, n in enumerate(D['names']) if n not in T['exclude']]
    ok = ~np.isnan(y) & (D['final'] == 1)
    tr = ok & (s >= TRAIN[0]) & (s <= TRAIN[1])
    va = ok & (s >= VAL[0]) & (s <= VAL[1])
    X = D['X'][:, feats]
    mu = np.nanmean(X[tr], axis=0); sd = np.nanstd(X[tr], axis=0); sd[sd < 1e-9] = 1.0
    Z = np.clip(np.nan_to_num((X - mu) / sd), -6, 6)
    P = dict(task=task, y=y, tr=tr, va=va, Z=Z, ZT=np.ascontiguousarray(Z.T), names=[D['names'][j] for j in feats],
             mu=mu, sd=sd, season=s, fold=s % 4, base=None)
    P['spread_raw'] = D['X'][:, D['names'].index('spread')]; P['total_raw'] = D['X'][:, D['names'].index('total_line')]
    P['has_line'] = D['X'][:, D['names'].index('has_line')]
    if base:
        P['base'], P['base_C'] = (base_ridge if is_reg(task) else base_logit)(Z, y, tr, s % 4)
    return P

def base_ridge(Z, y, fitmask, fold, alphas=(300, 1000, 3000, 10000, 30000)):
    from sklearn.linear_model import Ridge
    Zf, yf, ff = Z[fitmask], y[fitmask], fold[fitmask]
    best = None
    for a in alphas:
        oof = np.zeros(len(yf))
        for f in range(4):
            m = ff == f
            oof[m] = Ridge(alpha=a).fit(Zf[~m], yf[~m]).predict(Zf[m])
        mse = float(np.mean((oof - yf) ** 2))
        if best is None or mse < best[0]: best = (mse, a, oof)
    full = Ridge(alpha=best[1]).fit(Zf, yf).predict(Z)
    col = full.copy(); col[fitmask] = best[2]
    return col, best[1]

def lsq(G, y, ridge=1e-3):
    R = ridge * np.eye(G.shape[1]); R[0, 0] = 0
    return np.linalg.solve(G.T @ G + R, G.T @ y)

def base_logit(Z, y, fitmask, fold, Cs=(0.0003, 0.001, 0.003, 0.01)):
    """Ridge-logistic on every feature: out-of-fold values on fitmask rows, full-fit values everywhere else."""
    from sklearn.linear_model import LogisticRegression
    Zf, yf, ff = Z[fitmask], y[fitmask], fold[fitmask]
    best = None
    for C in Cs:
        oof = np.zeros(len(yf))
        for f in range(4):
            m = ff == f
            oof[m] = LogisticRegression(C=C, max_iter=3000).fit(Zf[~m], yf[~m]).decision_function(Zf[m])
        ll = logloss(oof, yf)
        if best is None or ll < best[0]: best = (ll, C, oof)
    full = LogisticRegression(C=best[1], max_iter=3000).fit(Zf, yf).decision_function(Z)
    col = full.copy(); col[fitmask] = best[2]
    return col, best[1]

# ------------------------------------------------------------------ trees
ARITY = {'add': 2, 'sub': 2, 'mul': 2, 'aq': 2, 'max': 2, 'min': 2, 'ifgt': 4, 'neg': 1, 'abs': 1, 'tanh': 1, 'sq': 1, 'gt0': 1}
OPS2 = ['add', 'sub', 'mul', 'aq', 'max', 'min']
OPS1 = ['neg', 'abs', 'tanh', 'sq', 'gt0']
OPW = {'add': 3, 'sub': 3, 'mul': 3, 'aq': 1.5, 'max': 1, 'min': 1, 'ifgt': 1.2, 'neg': 0.5, 'abs': 0.6, 'tanh': 1.2, 'sq': 0.6, 'gt0': 0.8}
OPLIST = list(OPW); OPPROB = np.array([OPW[o] for o in OPLIST]); OPPROB /= OPPROB.sum()

def ev(tree, i, ZT):
    k, v = tree[i]
    if k == 'f':
        return ZT[v], i + 1
    if k == 'c':
        return v, i + 1
    a, j = ev(tree, i + 1, ZT)
    if v in ('neg', 'abs', 'tanh', 'sq', 'gt0'):
        if v == 'neg': return -a, j
        if v == 'abs': return np.abs(a), j
        if v == 'tanh': return np.tanh(a), j
        if v == 'sq': return np.minimum(a * a, 36.0), j
        return (np.asarray(a) > 0) * 1.0, j
    b, j = ev(tree, j, ZT)
    if v == 'add': return a + b, j
    if v == 'sub': return a - b, j
    if v == 'mul': return np.clip(a * b, -1e4, 1e4), j
    if v == 'aq': return a / np.sqrt(1.0 + b * b), j
    if v == 'max': return np.maximum(a, b), j
    if v == 'min': return np.minimum(a, b), j
    c, j = ev(tree, j, ZT)
    d, j = ev(tree, j, ZT)
    return np.where(np.asarray(a) > b, c, d), j

def evaluate(tree, ZT):
    out, _ = ev(tree, 0, ZT)
    n = ZT.shape[1]
    out = np.broadcast_to(np.asarray(out, dtype=float), (n,))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def subtree_end(tree, i):
    need = 1
    while need:
        k, v = tree[i]
        need += (ARITY[v] if k == 'o' else 0) - 1
        i += 1
    return i

def height(tree, i=0):
    k, v = tree[i]
    if k != 'o':
        return 1, i + 1
    h = 0; j = i + 1
    for _ in range(ARITY[v]):
        hh, j = height(tree, j); h = max(h, hh)
    return h + 1, j

class Gen:
    def __init__(self, nfeat, rng, feat_pool=None, maxd=6, maxn=31):
        self.nfeat, self.rng, self.maxd, self.maxn = nfeat, rng, maxd, maxn
        self.pool = feat_pool if feat_pool is not None else list(range(nfeat))

    def term(self):
        if self.rng.random() < 0.2:
            return ('c', round(self.rng.uniform(-2, 2), 3))
        return ('f', self.rng.choice(self.pool))

    def op(self):
        return ('o', OPLIST[np.searchsorted(np.cumsum(OPPROB), self.rng.random())])

    def _rand(self, depth, full=False):
        if depth <= 1 or (not full and self.rng.random() < 0.3):
            return [self.term()]
        o = self.op()
        out = [o]
        for _ in range(ARITY[o[1]]):
            out += self._rand(depth - 1, full)
        return out

    def rand_tree(self, depth, full=False):
        for _ in range(30):                      # respect the size and depth limits from the start
            t = self._rand(depth, full)
            if self.ok(t):
                return t
        return [self.term()]

    def ok(self, t):
        return len(t) <= self.maxn and height(t)[0] <= self.maxd

    def mutate_subtree(self, t):
        i = self.rng.randrange(len(t)); e = subtree_end(t, i)
        return t[:i] + self.rand_tree(self.rng.randint(1, 3)) + t[e:]

    def mutate_point(self, t):
        t = list(t); i = self.rng.randrange(len(t)); k, v = t[i]
        if k == 'o':
            same = [o for o in OPLIST if ARITY[o] == ARITY[v] and o != v]
            if same: t[i] = ('o', self.rng.choice(same))
        else:
            t[i] = self.term()
        return t

    def jitter(self, t):
        return [('c', round(v + self.rng.gauss(0, 0.3), 3)) if k == 'c' else (k, v) for k, v in t]

    def hoist(self, t):
        i = self.rng.randrange(len(t)); return t[i:subtree_end(t, i)]

    def crossover(self, a, b):
        i = self.rng.randrange(len(a)); ea = subtree_end(a, i)
        j = self.rng.randrange(len(b)); eb = subtree_end(b, j)
        return a[:i] + b[j:eb] + a[ea:]

def tree_str(t, names, i=0):
    k, v = t[i]
    if k == 'f': return names[v], i + 1
    if k == 'c': return repr(v), i + 1
    args = []; j = i + 1
    for _ in range(ARITY[v]):
        s, j = tree_str(t, names, j); args.append(s)
    sym = {'add': '+', 'sub': '-', 'mul': '*'}
    if v in sym: return f'({args[0]} {sym[v]} {args[1]})', j
    return f'{v}({", ".join(args)})', j

def ind_key(ind):
    return '|'.join(','.join(f'{k}{v}' for k, v in g) for g in ind)

# ------------------------------------------------------------------ fitting
def fit_logit(G, y, ridge=1e-2, iters=8):
    beta = np.zeros(G.shape[1])
    for _ in range(iters):
        z = np.clip(G @ beta, -30, 30); p = 1 / (1 + np.exp(-z)); W = p * (1 - p) + 1e-9
        H = (G.T * W) @ G + ridge * np.eye(G.shape[1]); H[0, 0] -= ridge
        g = G.T @ (y - p) - ridge * np.r_[0, beta[1:]]
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-7: break
    return beta

def logloss(z, y):
    z = np.clip(z, -30, 30)
    return float(np.mean(np.logaddexp(0, z) - y * z))

def gene_matrix(ind, ZT, base=None):
    cols = [evaluate(g, ZT) for g in ind]
    if base is not None:
        cols = [base] + cols
    return np.column_stack([np.ones(ZT.shape[1])] + cols)

class Scorer:
    def __init__(self, P, lam=1e-5):
        self.P, self.lam = P, lam
        tr = P['tr']
        self.ZT = np.ascontiguousarray(P['ZT'][:, tr])
        self.y = P['y'][tr]
        self.bcol = None if P['base'] is None else P['base'][tr]
        self.fold = P['fold'][tr]
        self.cache = {}
        self.reg = is_reg(P['task'])
        self.var = float(np.var(self.y))
        self.base = 1.0 if self.reg else logloss(np.full(len(self.y), math.log(self.y.mean() / (1 - self.y.mean()))), self.y)

    def score(self, ind):
        key = ind_key(ind)
        if key in self.cache: return self.cache[key]
        G = gene_matrix(ind, self.ZT, self.bcol)
        sd = G[:, 1:].std(axis=0)
        if np.any(sd < 1e-9) or not np.all(np.isfinite(G)):
            self.cache[key] = (9.0, 0.0); return self.cache[key]
        mu = G[:, 1:].mean(axis=0)
        G[:, 1:] = (G[:, 1:] - mu) / sd
        oof = np.zeros(len(self.y))
        for f in range(4):
            m = self.fold == f
            b = lsq(G[~m], self.y[~m]) if self.reg else fit_logit(G[~m], self.y[~m])
            oof[m] = G[m] @ b
        if self.reg:
            ll = float(np.mean((oof - self.y) ** 2)) / self.var
            acc = float(np.mean(np.abs(oof - self.y)))          # MAE for regression tasks
        else:
            ll = logloss(oof, self.y)
            acc = float(np.mean((oof > 0) == (self.y > 0.5)))
        size = sum(len(g) for g in ind)
        fit = ll + self.lam * size
        self.cache[key] = (fit, acc)
        return self.cache[key]

def fit_full(ind, P, rows):
    """Fit the combination layer on `rows` (boolean mask) and return (beta, mu, sd)."""
    G = gene_matrix(ind, P['ZT'], P['base'])
    mu = G[rows, 1:].mean(axis=0); sd = G[rows, 1:].std(axis=0); sd[sd < 1e-9] = 1.0
    G[:, 1:] = (G[:, 1:] - mu) / sd
    b = lsq(G[rows], P['y'][rows]) if is_reg(P['task']) else fit_logit(G[rows], P['y'][rows])
    return b, mu, sd, G @ b

def metrics(z, y, m, P=None):
    if P is not None and is_reg(P['task']):
        zz, yy = z[m], y[m]
        out = dict(n=int(m.sum()), mae=round(float(np.mean(np.abs(zz - yy))), 4), logloss=round(float(np.mean((zz - yy) ** 2)), 3))
        line = (P['spread_raw'] if P['task'] == 'margin' else P['total_raw'])[m]
        ok = (P['has_line'][m] > 0) & (yy != line)
        out['line_mae'] = round(float(np.mean(np.abs(line - yy))), 4)
        out['acc'] = round(float(np.mean(((zz - line) > 0)[ok] == ((yy - line) > 0)[ok])), 4)   # side vs the line
        if P['task'] == 'margin':
            out['su_acc'] = round(float(np.mean((zz > 0) == (yy > 0))), 4)
        out['acc_top10'] = 0.0
        return out
    z, y = z[m], y[m]
    acc = float(np.mean((z > 0) == (y > 0.5)))
    out = dict(n=int(m.sum()), acc=round(acc, 4), logloss=round(logloss(z, y), 5))
    order = np.argsort(-np.abs(z))
    for q in (0.2, 0.1):
        k = max(1, int(len(z) * q)); sel = order[:k]
        out[f'acc_top{int(q * 100)}'] = round(float(np.mean((z[sel] > 0) == (y[sel] > 0.5))), 4)
    return out

# ------------------------------------------------------------------ evolution
def run(task, seed, gens, pop, islands, null=False, lam=1e-5, subset=0.7, log_every=10, time_limit=None, base=False, maxn=31, maxd=6, maxgenes=4):
    D = load(); P = prepare(D, task, null_seed=(1000 + seed) if null else None, base=base)
    rng = random.Random(seed); S = Scorer(P, lam=lam)
    nf = len(P['names'])
    gens_by_island = []
    for k in range(islands):
        pool = list(range(nf)) if k == 0 else rng.sample(range(nf), max(8, int(nf * subset)))
        gens_by_island.append(Gen(nf, random.Random(seed * 100 + k), pool, maxd=maxd, maxn=maxn))
    def rand_ind(G):
        return [G.rand_tree(G.rng.randint(2, min(5, maxd)), full=G.rng.random() < 0.5) for _ in range(G.rng.randint(1, 3))]
    popn = [[rand_ind(G) for _ in range(pop)] for G in gens_by_island]
    hof = {}
    t0 = time.time(); logrows = []
    for gen in range(gens):
        for k in range(islands):
            G = gens_by_island[k]
            scored = [(S.score(ind), ind) for ind in popn[k]]
            for (fit, acc), ind in scored:
                key = ind_key(ind)
                if key not in hof: hof[key] = (fit, acc, ind)
            scored.sort(key=lambda x: x[0][0])
            elite = [ind for _, ind in scored[:4]]
            def tourney():
                c = G.rng.sample(scored, 6); return min(c, key=lambda x: x[0][0])[1]
            nxt = list(elite)
            while len(nxt) < pop:
                a = tourney(); r = G.rng.random()
                child = [list(g) for g in a]
                if r < 0.15:           # gene-level crossover
                    b = tourney(); gi = G.rng.randrange(len(child)); child[gi] = list(G.rng.choice(b))
                elif r < 0.55:         # subtree crossover inside a gene
                    b = tourney(); gi = G.rng.randrange(len(child)); new = G.crossover(child[gi], G.rng.choice(b))
                    if G.ok(new): child[gi] = new
                elif r < 0.70:
                    gi = G.rng.randrange(len(child)); new = G.mutate_subtree(child[gi])
                    if G.ok(new): child[gi] = new
                elif r < 0.80:
                    gi = G.rng.randrange(len(child)); child[gi] = G.mutate_point(child[gi])
                elif r < 0.87:
                    gi = G.rng.randrange(len(child)); child[gi] = G.jitter(child[gi])
                elif r < 0.92:
                    gi = G.rng.randrange(len(child)); child[gi] = G.hoist(child[gi])
                elif r < 0.96 and len(child) < maxgenes:
                    child.append(G.rand_tree(G.rng.randint(2, min(4, maxd))))
                elif len(child) > 1:
                    child.pop(G.rng.randrange(len(child)))
                nxt.append(child)
            popn[k] = nxt
        if (gen + 1) % 10 == 0 and islands > 1:      # ring migration of the best few
            bests = [sorted(p, key=lambda ind: S.score(ind)[0])[:5] for p in popn]
            for k in range(islands):
                popn[(k + 1) % islands][-5:] = [[list(g) for g in ind] for ind in bests[k]]
        if (gen + 1) % log_every == 0 or gen == 0:
            best = min(hof.values(), key=lambda x: x[0])
            logrows.append(dict(gen=gen + 1, best_fit=round(best[0], 5), best_oof_acc=round(best[1], 4), evals=len(S.cache), secs=round(time.time() - t0)))
            print(f"[{task}{' NULL' if null else ''} s{seed}] gen {gen + 1}: best OOF logloss+pen {best[0]:.5f} (base {S.base:.5f}) OOF acc {best[1]:.4f} evals {len(S.cache)} {time.time() - t0:.0f}s", flush=True)
        if time_limit and time.time() - t0 > time_limit:
            break
    # ---- hall of fame -> validation
    top = sorted(hof.values(), key=lambda x: x[0])[:300]
    res = []
    for fit, acc, ind in top:
        b, mu, sd, z = fit_full(ind, P, P['tr'])
        res.append(dict(fit=round(fit, 6), oof_acc=round(acc, 4), size=sum(len(g) for g in ind),
                        train=metrics(z, P['y'], P['tr'], P), val=metrics(z, P['y'], P['va'], P),
                        genes=[tree_str(g, P['names'])[0] for g in ind], ind=[[list(x) for x in g] for g in ind]))
    by_val = sorted(res, key=lambda r: (r['val']['logloss'], r['size']))
    base_only = None
    if P['base'] is not None:
        b, mu, sd, z = fit_full([], P, P['tr'])
        base_only = dict(train=metrics(z, P['y'], P['tr'], P), val=metrics(z, P['y'], P['va'], P))
    if is_reg(task):
        base_val = float(np.mean((P['y'][P['va']] - P['y'][P['tr']].mean()) ** 2))
    else:
        base_val = logloss(np.full(P['va'].sum(), math.log(P['y'][P['tr']].mean() / (1 - P['y'][P['tr']].mean()))), P['y'][P['va']])
    return dict(task=task, desc=TASKS[task]['desc'], seed=seed, null=null, gens=gens, pop=pop, islands=islands, lam=lam,
                base=bool(base), base_C=P.get('base_C'), maxn=maxn, maxd=maxd, maxgenes=maxgenes,
                base_train=round(S.base, 5), base_val=round(base_val, 5), log=logrows, evals=len(S.cache),
                best_by_train=res[0], best_by_val=by_val[0], top_val=by_val[:25], names=P['names'], base_only=base_only)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gens', type=int, default=80); ap.add_argument('--pop', type=int, default=300)
    ap.add_argument('--islands', type=int, default=4); ap.add_argument('--lam', type=float, default=1e-5)
    ap.add_argument('--null', action='store_true'); ap.add_argument('--time', type=float, default=None)
    ap.add_argument('--base', action='store_true'); ap.add_argument('--maxn', type=int, default=31)
    ap.add_argument('--maxd', type=int, default=6); ap.add_argument('--maxgenes', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = run(a.task, a.seed, a.gens, a.pop, a.islands, null=a.null, lam=a.lam, time_limit=a.time, base=a.base,
              maxn=a.maxn, maxd=a.maxd, maxgenes=a.maxgenes)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh: json.dump(out, fh)
    bv, bt = out['best_by_val'], out['best_by_train']
    print(f"DONE {a.task}{' NULL' if a.null else ''} seed {a.seed}: best-by-train OOF acc {bt['oof_acc']} val acc {bt['val']['acc']} | "
          f"best-by-val val acc {bv['val']['acc']} val ll {bv['val']['logloss']} (base {out['base_val']}) size {bv['size']}")
    if is_reg(a.task):
        print(f"  val MAE {bv['val']['mae']} (line {bv['val']['line_mae']}), side vs line {bv['val']['acc']}")
    print('  genes:', ' ;; '.join(bv['genes']))
    if out['base_only']:
        print(f"  baseline alone: val acc {out['base_only']['val']['acc']} val ll {out['base_only']['val']['logloss']}")
