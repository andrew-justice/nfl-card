#!/usr/bin/env python3
"""
evalfam.py -- does each new feature family help?  Walk-forward validation on 2016-2020: every season is
predicted by models fitted on 2002 through the season before it.  2021-2025 is not touched here.
"""
import numpy as np, sys, os, warnings, json, itertools
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp
from sklearn.linear_model import Ridge, LogisticRegression

D = gp.load(); names = D['names']; s = D['season']; fin = D['final'] == 1
ym, yt, yw = D['Y']['y_margin'], D['Y']['y_total'], D['Y']['y_win']
col = lambda n: D['X'][:, names.index(n)]
spread, tline = col('spread'), col('total_line')
is_pb = lambda n: '_pb_' in n
is_inj = lambda n: 'inj' in n
is_wx = lambda n: n.startswith('wx_')
base = [n for n in names if n not in gp.MARKET and not (is_pb(n) or is_inj(n) or is_wx(n))]
fams = {'base': base,
        '+pbp': base + [n for n in names if is_pb(n)],
        '+injuries': base + [n for n in names if is_inj(n)],
        '+weather': base + [n for n in names if is_wx(n)],
        'all': [n for n in names if n not in gp.MARKET]}
VAL = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 2016, (int(sys.argv[2]) if len(sys.argv) > 2 else 2020) + 1))
Cs, As, At = (0.0005, 0.001, 0.002), (2000, 3000, 5000, 8000), (3000, 5000, 8000, 12000)

def walk(feats):
    idx = [names.index(n) for n in feats]; X = D['X'][:, idx]
    out = {('w', c): np.full(len(s), np.nan) for c in Cs}
    out.update({('m', a): np.full(len(s), np.nan) for a in As}); out.update({('t', a): np.full(len(s), np.nan) for a in At})
    for T in VAL:
        tr = fin & (s >= 2002) & (s < T); te = s == T
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd[sd < 1e-9] = 1
        Z = np.clip(np.nan_to_num((X - mu) / sd), -6, 6)
        ok = tr & ~np.isnan(yw)
        for c in Cs: out[('w', c)][te] = LogisticRegression(C=c, max_iter=5000).fit(Z[ok], yw[ok]).decision_function(Z[te])
        for a in As: out[('m', a)][te] = Ridge(alpha=a).fit(Z[tr], ym[tr]).predict(Z[te])
        for a in At: out[('t', a)][te] = Ridge(alpha=a).fit(Z[tr], yt[tr]).predict(Z[te])
    return out

vm = fin & np.isin(s, VAL)
res = {}
for fam, feats in fams.items():
    P = walk(feats)
    okw = vm & ~np.isnan(yw)
    best_w = min(Cs, key=lambda c: np.mean(np.logaddexp(0, P[('w', c)][okw]) - yw[okw] * P[('w', c)][okw]))
    zw = P[('w', best_w)]
    ll = float(np.mean(np.logaddexp(0, zw[okw]) - yw[okw] * zw[okw])); acc = float(np.mean((zw[okw] > 0) == (yw[okw] > 0.5)))
    best_m = min(As, key=lambda a: np.mean(np.abs(P[('m', a)][vm] - ym[vm]))); pm = P[('m', best_m)]
    best_t = min(At, key=lambda a: np.mean(np.abs(P[('t', a)][vm] - yt[vm]))); pt = P[('t', best_t)]
    mae_m = float(np.mean(np.abs(pm[vm] - ym[vm]))); mae_t = float(np.mean(np.abs(pt[vm] - yt[vm])))
    q = vm & (ym != spread); ats = float(np.mean(((pm - spread) > 0)[q] == (ym > spread)[q]))
    q4 = q & (np.abs(pm - spread) >= 4); ats4 = (int(np.sum(((pm - spread) > 0)[q4] == (ym > spread)[q4])), int(q4.sum()))
    q = vm & (yt != tline); ou = float(np.mean(((pt - tline) > 0)[q] == (yt > tline)[q]))
    res[fam] = dict(n_feat=len(feats), win_acc=round(acc, 4), win_logloss=round(ll, 5), C=best_w, margin_mae=round(mae_m, 3), alpha_m=best_m,
                    total_mae=round(mae_t, 3), alpha_t=best_t, ats=round(ats, 4), ats4=ats4, ou=round(ou, 4))
    print(f"{fam:10s} {len(feats):3d} feats | winners {acc:.4f} (logloss {ll:.5f}, C {best_w}) | spread MAE {mae_m:.3f} (a {best_m}) | "
          f"total MAE {mae_t:.3f} (a {best_t}) | ATS {ats:.3f}, 4+ {ats4[0]}-{ats4[1]-ats4[0]} | O/U {ou:.3f}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'evalfam-{VAL[0]}-{VAL[-1]}.json'), 'w'), indent=1)
