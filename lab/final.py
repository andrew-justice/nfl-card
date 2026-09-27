#!/usr/bin/env python3
"""
final.py -- walk-forward fit of the selected models, the one-time test, and the export for nfl-build.py.

For every season T from 2010 to 2026: standardize on and fit to seasons 2002..T-1 only, predict season T.
2021-2025 is the locked test (never used for any choice); 2026 is live.
Exports lab/model-export.json: production coefficients (fit on 2002-2025), residuals, calibration, backtest.
"""
import numpy as np, json, os, sys, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp
from sklearn.linear_model import Ridge, LogisticRegression

LAB = os.path.dirname(os.path.abspath(__file__))
SPEC = dict(win=dict(C=0.001, features='all'), margin=dict(alpha=3000.0, features='all'), total=dict(alpha=5000.0, features='base+weather'))
V3_NEW = lambda n: ('_pb_' in n) or ('inj' in n) or n.startswith('wx_')

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - r) / d, 4), round((c + r) / d, 4)]

def main():
    D = gp.load(); names = D['names']; s = D['season']; fin = D['final'] == 1
    cols = [j for j, n in enumerate(names) if n not in gp.MARKET]
    fnames = [names[j] for j in cols]
    X = D['X'][:, cols]
    col = lambda n: D['X'][:, names.index(n)]
    ym, yt, yw = D['Y']['y_margin'], D['Y']['y_total'], D['Y']['y_win']
    P = dict(margin=np.full(len(s), np.nan), total=np.full(len(s), np.nan), winlogit=np.full(len(s), np.nan))
    tot_idx = [i for i, n in enumerate(fnames) if not V3_NEW(n) or n.startswith('wx_')]    # totals: base features plus weather
    prod = None
    for T in range(2010, 2027):
        tr = fin & (s >= 2002) & (s < T)
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd[sd < 1e-9] = 1.0
        Z = np.clip(np.nan_to_num((X - mu) / sd), -6, 6)
        rm = Ridge(alpha=SPEC['margin']['alpha']).fit(Z[tr], ym[tr])
        rt = Ridge(alpha=SPEC['total']['alpha']).fit(Z[tr][:, tot_idx], yt[tr])
        okw = tr & ~np.isnan(yw)
        lw = LogisticRegression(C=SPEC['win']['C'], max_iter=5000).fit(Z[okw], yw[okw])
        te = s == T
        P['margin'][te] = rm.predict(Z[te]); P['total'][te] = rt.predict(Z[te][:, tot_idx]); P['winlogit'][te] = lw.decision_function(Z[te])
        if T == 2026:
            prod = dict(features=fnames, mu=[float(x) for x in mu], sd=[float(x) for x in sd],
                        margin=dict(intercept=float(rm.intercept_), coef=[float(x) for x in rm.coef_]),
                        total=dict(intercept=float(rt.intercept_), coef=[float(rt.coef_[tot_idx.index(i)]) if i in tot_idx else 0.0 for i in range(len(fnames))]),
                        win=dict(intercept=float(lw.intercept_[0]), coef=[float(x) for x in lw.coef_[0]]),
                        fit_seasons=[2002, 2025], clip=6.0, spec=SPEC)
    spread, tline, ml = col('spread'), col('total_line'), col('ml_logit')
    fav = np.where(spread != 0, spread, ml)
    old_m, old_t = col('pred'), col('pred_total')

    def su(pred, m):
        m = m & ~np.isnan(yw)
        k = int(((pred[m] > 0) == (yw[m] > 0.5)).sum()); n = int(m.sum())
        return dict(n=n, wins=k, acc=round(k / n, 4), ci95=wilson(k, n))
    def side(pred, line, y, m, th=0.0):
        m = m & ~np.isnan(line) & (y != line) & (np.abs(pred - line) >= th) & ~np.isnan(y)
        k = int(((pred[m] - line[m] > 0) == (y[m] - line[m] > 0)).sum()); n = int(m.sum())
        return [k, n - k]
    def mae(pred, y, m):
        m = m & ~np.isnan(y); return round(float(np.mean(np.abs(pred[m] - y[m]))), 3)

    blocks = {'2010-2015': (2010, 2015), '2016-2020 (validation)': (2016, 2020), '2021-2025 (locked test)': (2021, 2025), '2026 to date': (2026, 2026)}
    rep = {}
    for lab, (a, b) in blocks.items():
        m = fin & (s >= a) & (s <= b)
        rep[lab] = dict(
            su_new=su(P['winlogit'], m), su_new_margin=su(P['margin'], m), su_old=su(old_m, m), su_market=su(fav, m),
            mae_new=mae(P['margin'], ym, m), mae_old=mae(old_m, ym, m), mae_line=mae(spread, ym, m),
            tmae_new=mae(P['total'], yt, m), tmae_old=mae(old_t, yt, m), tmae_line=mae(tline, yt, m),
            ats_new={th: side(P['margin'], spread, ym, m, th) for th in (0, 1, 2, 3, 4)},
            ats_old={th: side(old_m, spread, ym, m, th) for th in (0, 2)},
            ou_new={th: side(P['total'], tline, yt, m, th) for th in (0, 1, 2, 3, 4)},
            ou_old={th: side(old_t, tline, yt, m, th) for th in (0, 2)})
    print(json.dumps(rep, indent=1))
    json.dump(dict(rep=rep, pred=dict(game_id=D['game_id'], season=s.tolist(), margin=np.nan_to_num(P['margin'], nan=-999).tolist(),
                                      total=np.nan_to_num(P['total'], nan=-999).tolist(), winlogit=np.nan_to_num(P['winlogit'], nan=-999).tolist()),
                   prod=prod), open(os.path.join(LAB, 'final-walkforward.json'), 'w'))

if __name__ == '__main__':
    main()
