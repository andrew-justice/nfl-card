#!/usr/bin/env python3
"""
keys.py -- how likely each exact margin and total is, given the market's number.

Used to price lines that differ by half points (line shopping, buying points, grading closing-line value):
  P(margin = k | center s)  is proportional to  Normal(k; s, sd_m) x w_m[|k|]
  P(total  = k | center t)  is proportional to  Normal(k; t, sd_t(t)) x w_t[k],   sd_t(t) = a + b (t - 44)
The weights capture key numbers (3, 7, 6, 10, 14 for margins; 37, 41, 44, 47, 51 ... for totals) that a smooth
curve misses. Fitted on 2015 onward, the seasons played with the longer extra point.

  python3 keys.py            validate (fit 2015-22, score 2023-25) and write the final fit (2015-25) into ../data/model.json
"""
import csv, json, os, sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar

LAB = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(LAB, '..', 'nfl-model', 'cache', 'games.csv')
if not os.path.exists(GAMES):
    GAMES = os.path.join(LAB, '..', 'cache', 'games.csv')
MODEL = os.path.join(LAB, '..', 'nfl-model', 'data', 'model.json')
if not os.path.exists(MODEL):
    MODEL = os.path.join(LAB, '..', 'data', 'model.json')

MG = np.arange(-80, 81)          # margin grid
TG = np.arange(0, 141)           # total grid
M_MAX, T_MAX = 40, 110           # weights estimated up to these; beyond, the smooth curve alone
SHRINK = 15.0                    # log-weight shrinkage: factor O/(O+SHRINK)

def load():
    rows = [r for r in csv.DictReader(open(GAMES)) if r['result'] and r['spread_line'] and r['total_line']]
    f = lambda k: np.array([float(r[k]) for r in rows])
    return dict(season=np.array([int(r['season']) for r in rows]), s=f('spread_line'), R=f('result'), t=f('total_line'), T=f('total'))

def smooth(center, sd, grid):
    c, d = center[:, None], sd[:, None]
    return norm.cdf((grid[None, :] + 0.5 - c) / d) - norm.cdf((grid[None, :] - 0.5 - c) / d)

def probs(S, w, idx):
    P = S * w[idx][None, :]
    return P / P.sum(1, keepdims=True)

def fit_weights(S, obs_idx, idx, n_w, cap, iters=200, exact=()):
    """IPF to the maximum-likelihood weights, then shrink each log-weight toward 0 by O/(O+SHRINK).
    Indices in `exact` keep their maximum-likelihood weight (a tie is rare for structural reasons, not by chance)."""
    w = np.ones(n_w)
    O = np.bincount(obs_idx, minlength=n_w).astype(float)
    free = np.arange(n_w) <= cap
    for _ in range(iters):
        E = np.bincount(np.broadcast_to(idx, S.shape).ravel(), weights=probs(S, w, idx).ravel(), minlength=n_w)
        upd = np.where(free & (E > 0), (O + 1e-9) / np.maximum(E, 1e-12), 1.0)
        w = w * upd
        w = w / np.exp(np.mean(np.log(w[free & (O > 0)])))
    keep = np.log(np.maximum(w, 1e-6))
    lw = keep * (O / (O + SHRINK))
    for i in exact:
        lw[i] = keep[i]
    lw[~free] = 0.0
    return np.exp(lw)

def nll(P, obs_col):
    return -np.mean(np.log(np.maximum(P[np.arange(len(obs_col)), obs_col], 1e-12)))

def fit_margin(d, m):
    s, R = d['s'][m], d['R'][m]
    idx = np.abs(MG)
    obs_col = (R - MG[0]).astype(int); obs_idx = np.abs(R).astype(int)
    delta, sd, w = 0.0, 13.0, np.ones(81)
    for _ in range(4):
        def f(p):
            S = smooth(s + p[0], np.full(len(s), p[1]), MG)
            return nll(probs(S, w, idx), obs_col)
        r = minimize(f, [delta, sd], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7))
        delta, sd = r.x
        w = fit_weights(smooth(s + delta, np.full(len(s), sd), MG), obs_idx, idx, 81, M_MAX, exact=(0,))
    return dict(delta=float(delta), sd=float(sd), w=w)

def fit_total(d, m):
    t, T = d['t'][m], d['T'][m]
    idx = TG
    obs_col = T.astype(int); obs_idx = T.astype(int)
    a, b, delta, w = 13.2, 0.1, 0.0, np.ones(len(TG))
    for _ in range(4):
        def f(p):
            S = smooth(t + p[2], np.maximum(5, p[0] + p[1] * (t - 44)), TG)
            return nll(probs(S, w, idx), obs_col)
        r = minimize(f, [a, b, delta], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7))
        a, b, delta = r.x
        w = fit_weights(smooth(t + delta, np.maximum(5, a + b * (t - 44)), TG), obs_idx, idx, len(TG), T_MAX)
    return dict(a=float(a), b=float(b), delta=float(delta), w=w)

def margin_P(d, m, fm, use_w=True):
    s = d['s'][m]
    S = smooth(s + fm['delta'], np.full(len(s), fm['sd']), MG)
    return probs(S, fm['w'] if use_w else np.ones(81), np.abs(MG))

def total_P(d, m, ft, use_w=True):
    t = d['t'][m]
    S = smooth(t + ft['delta'], np.maximum(5, ft['a'] + ft['b'] * (t - 44)), TG)
    return probs(S, ft['w'] if use_w else np.ones(len(TG)), TG)

def validate(d):
    tr = (d['season'] >= 2015) & (d['season'] <= 2022)
    te = (d['season'] >= 2023) & (d['season'] <= 2025)
    old = (d['season'] >= 2002) & (d['season'] <= 2014)
    fm, ft = fit_margin(d, tr), fit_total(d, tr)
    fm_old = fit_margin(d, old)
    R, T = d['R'][te], d['T'][te]
    oc_m = (R - MG[0]).astype(int); oc_t = T.astype(int)
    Pm, Pm0, Pm_old = margin_P(d, te, fm), margin_P(d, te, fm, False), margin_P(d, te, fm_old)
    Pt, Pt0 = total_P(d, te, ft), total_P(d, te, ft, False)
    print(f"fit 2015-22: margin sd {fm['sd']:.2f}, shift {fm['delta']:+.2f}; total sd {ft['a']:.2f}{ft['b']:+.3f}(t-44), shift {ft['delta']:+.2f}")
    print(f"held-out 2023-25, {te.sum()} games: log loss per game (lower is better)")
    print(f"  margin: smooth only {nll(Pm0, oc_m):.4f} | with key weights {nll(Pm, oc_m):.4f} | weights from 2002-14 {nll(Pm_old, oc_m):.4f}")
    print(f"  total:  smooth only {nll(Pt0, oc_t):.4f} | with key weights {nll(Pt, oc_t):.4f}")
    print("  exact margins, predicted vs actual count (either team):")
    for k in (1, 2, 3, 4, 6, 7, 8, 10, 14):
        cols = [k - MG[0], -k - MG[0]]
        print(f"    {k:2d}: weighted {Pm[:, cols].sum():6.1f}  smooth {Pm0[:, cols].sum():6.1f}  actual {int(np.sum(np.abs(R) == k))}")
    # the money question: how often does a line land exactly on the number?
    for lo, hi, k in ((2.5, 3.5, 3), (6.5, 7.5, 7), (-3.5, -2.5, -3), (-7.5, -6.5, -7)):
        q = (d['s'][te] >= lo) & (d['s'][te] <= hi)
        print(f"    lines {lo:+.1f}..{hi:+.1f} ({q.sum()} games): P(margin={k:+d}) predicted {Pm[q][:, k - MG[0]].mean():.3f} (smooth {Pm0[q][:, k - MG[0]].mean():.3f}), actual {np.mean(R[q] == k):.3f}")
    for k in (37, 41, 43, 44, 47, 51):
        print(f"    total {k}: weighted {Pt[:, k].sum():5.1f}  smooth {Pt0[:, k].sum():5.1f}  actual {int(np.sum(T == k))}")
    return dict(margin_ll=[nll(Pm0, oc_m), nll(Pm, oc_m)], total_ll=[nll(Pt0, oc_t), nll(Pt, oc_t)])

def export(d):
    m = (d['season'] >= 2015) & (d['season'] <= 2025)
    fm, ft = fit_margin(d, m), fit_total(d, m)
    out = dict(fit='2015-2025', n=int(m.sum()),
               margin=dict(sd=round(fm['sd'], 3), w=[round(float(x), 4) for x in fm['w'][:M_MAX + 1]]),
               total=dict(a=round(ft['a'], 3), b=round(ft['b'], 4), w=[round(float(x), 4) for x in ft['w'][:T_MAX + 1]]))
    print(f"final fit 2015-25 ({out['n']} games): margin sd {out['margin']['sd']}, total sd {out['total']['a']} {out['total']['b']:+}(t-44)")
    print("  margin weights 0-17:", ' '.join(f'{k}:{w:.2f}' for k, w in enumerate(out['margin']['w'][:18])))
    return out

if __name__ == '__main__':
    d = load()
    validate(d)
    keys = export(d)
    M = json.load(open(MODEL))
    M['keys'] = keys
    json.dump(M, open(MODEL, 'w'), separators=(',', ':'))
    print('wrote keys into', os.path.normpath(MODEL))
