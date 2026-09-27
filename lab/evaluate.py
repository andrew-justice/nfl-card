#!/usr/bin/env python3
"""
evaluate.py -- the one-time test.  For each chosen model: refit the combination layer on 2002-2020,
then score 2021-2025 (locked test) and 2026 to date.  Nothing here feeds back into the search.

usage: python3 evaluate.py runs/su_ind-final.json [more.json ...]
A json may hold a single individual under best_by_val, or an ensemble under 'members'.
"""
import numpy as np, json, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - r) / d, 4), round((c + r) / d, 4)

def binom_p_above(k, n, p0):
    """one-sided P(X >= k) under Binomial(n, p0), normal approximation with continuity correction"""
    if n == 0: return 1.0
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    zz = (k - 0.5 - mu) / sd
    return round(0.5 * math.erfc(zz / math.sqrt(2)), 4)

def score_members(members, D, task):
    P = gp.prepare(D, task)
    s = P['season']; fin = D['final'] == 1
    fitrows = P['tr'] | P['va']                      # 2002-2020
    zs = []
    for ind in members:
        ind = [[tuple(x) for x in g] for g in ind]
        b, mu, sd, z = gp.fit_full(ind, P, fitrows)
        zs.append(z)
    z = np.mean(zs, axis=0)
    return P, z

def block(z, y, m, extra=None):
    m = m & ~np.isnan(y)
    if m.sum() == 0: return dict(n=0)
    hit = (z[m] > 0) == (y[m] > 0.5); k = int(hit.sum()); n = int(m.sum())
    out = dict(n=n, wins=k, acc=round(k / n, 4), ci95=wilson(k, n))
    order = np.argsort(-np.abs(z[m]))
    for q in (0.2, 0.1):
        sel = order[:max(1, int(n * q))]; kk = int(hit[sel].sum()); nn = len(sel)
        out[f'top{int(q * 100)}'] = dict(n=nn, wins=kk, acc=round(kk / nn, 4), ci95=wilson(kk, nn))
    return out

def main(paths):
    D = gp.load()
    names = D['names']; col = lambda n: D['X'][:, names.index(n)]
    s = D['season']; fin = D['final'] == 1
    test = fin & (s >= 2021) & (s <= 2025); live = fin & (s == 2026)
    report = {}
    for path in paths:
        R = json.load(open(path))
        task = R['task']
        members = R.get('members') or [R['best_by_val']['ind']]
        P, z = score_members(members, D, task)
        y = P['y']
        rep = dict(task=task, source=os.path.basename(path), members=len(members),
                   test=block(z, y, test), live=block(z, y, live))
        if task.startswith('su'):
            spread, ml = col('spread'), col('ml_logit'); fav = np.where(spread != 0, spread, ml)
            rep['market_test'] = block(fav, y, test); rep['market_live'] = block(fav, y, live)
            rep['current_test'] = block(col('pred'), y, test); rep['current_live'] = block(col('pred'), y, live)
        else:
            for key in ('test', 'live'):
                b = rep[key]
                if b.get('n'):
                    b['p_vs_50'] = binom_p_above(b['wins'], b['n'], 0.5)
                    b['p_vs_breakeven'] = binom_p_above(b['wins'], b['n'], 0.5238)
                    b['roi_at_110'] = round((b['wins'] * (100 / 110) - (b['n'] - b['wins'])) / b['n'], 4)
        report[path] = rep
        print(json.dumps(rep, indent=1))
    return report

if __name__ == '__main__':
    main(sys.argv[1:])
