#!/usr/bin/env python3
"""
export.py -- package the selected models for nfl-build.py: coefficients, residuals, calibration,
walk-forward backtest summaries, feature groups for explanations, and a summary of the genetic search.
Reads final-walkforward.json (from final.py) and runs/*.json.  Writes ../nfl-model/data/model.json.
"""
import numpy as np, json, os, sys, glob, math, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp

LAB = os.path.dirname(os.path.abspath(__file__))
_up = os.path.dirname(LAB)
MODEL = _up if os.path.exists(os.path.join(_up, 'nfl-build.py')) else os.path.join(_up, 'nfl-model')
OUT = os.path.join(MODEL, 'data', 'model.json')

def group_of(n):
    if n.startswith('wx_'):
        return 'Weather'
    if 'inj' in n:
        return 'Injuries'
    if '_pb_st' in n:
        return 'Special teams and penalties'
    if '_pb_' in n:
        return 'Efficiency'
    if n in ('elo_spread', 'elo_h', 'elo_a', 'qa_h', 'qa_a', 'qa_diff', 'pred', 'pred_total', 'has_pred', 'qb_new_h', 'qb_new_a'):
        return 'Ratings and QB'
    if any(k in n for k in ('qb_starts', 'qb_rookie', 'coach_')):
        return 'QB and coach experience'
    if any(k in n for k in ('_ats', '_ou_', 'ats_streak')) or n.endswith('_ou_m') or n.endswith('_ou_l'):
        return 'Record against the line'
    if any(k in n for k in ('to_margin', 'give', 'take', 'fum_luck', 'luck', 'fg_luck', 'last_to')):
        return 'Turnovers and luck'
    if any(k in n for k in ('sack', 'qbhit')):
        return 'Pass rush and protection'
    if any(k in n for k in ('ret_margin', 'punt_net', 'pen_margin')):
        return 'Special teams and penalties'
    if n in ('epa_net', 'oh', 'dh', 'oa', 'da') or any(k in n for k in ('epa_off', 'epa_def', 'epa_pg', 'pass_off', 'pass_def', 'rush_off', 'rush_def', 'fd_', 'expl_', 'cpoe', 'last_epa')):
        return 'Efficiency'
    if any(k in n for k in ('margin', 'win', 'pts_', 'wpct', '_pd', '_gp', 'streak')):
        return 'Recent results'
    return 'Schedule, rest and venue'

def fit_logit2(x, y, iters=50):
    a = b = 0.0
    for _ in range(iters):
        z = np.clip(a + b * x, -30, 30); p = 1 / (1 + np.exp(-z)); w = p * (1 - p)
        g0, g1 = np.sum(y - p), np.sum((y - p) * x)
        h00, h01, h11 = np.sum(w), np.sum(w * x), np.sum(w * x * x)
        det = h00 * h11 - h01 * h01
        da, db = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        a += da; b += db
        if abs(da) + abs(db) < 1e-10: break
    return float(a), float(b)

def main():
    W = json.load(open(os.path.join(LAB, 'final-walkforward.json')))
    D = gp.load(); names = D['names']; col = lambda n: D['X'][:, names.index(n)]
    s = D['season']; fin = D['final'] == 1
    pm = np.array(W['pred']['margin']); pt = np.array(W['pred']['total']); pw = np.array(W['pred']['winlogit'])
    have = pm > -998
    ym, yt, yw = D['Y']['y_margin'], D['Y']['y_total'], D['Y']['y_win']
    spread, tline, ml = col('spread'), col('total_line'), col('ml_logit')
    hl = col('has_line') > 0
    bt_rows = fin & have & (s >= 2010) & (s <= 2025)

    # residuals for probabilities (2014-2025 walk-forward), calibration (2010-2025)
    rm = bt_rows & (s >= 2014)
    resid = dict(margin=[round(float(x), 2) for x in (ym - pm)[rm]], total=[round(float(x), 2) for x in (yt - pt)[rm]])
    cs = bt_rows & hl & (ym != spread); ct = bt_rows & hl & (yt != tline)
    ca = fit_logit2((pm - spread)[cs], (ym > spread)[cs].astype(float))
    cb = fit_logit2((pt - tline)[ct], (yt > tline)[ct].astype(float))
    calib = dict(spread=[round(ca[0], 5), round(ca[1], 5)], total=[round(cb[0], 5), round(cb[1], 5)], n_spread=int(cs.sum()), n_total=int(ct.sum()))

    def block(m):
        m = m & have
        lined = m & hl
        o = dict(n=int(m.sum()), mae_model=round(float(np.mean(np.abs(pm[m] - ym[m]))), 2), mae_close=round(float(np.mean(np.abs(spread[lined] - ym[lined]))), 2),
                 mae_total_model=round(float(np.mean(np.abs(pt[m] - yt[m]))), 2), mae_total_close=round(float(np.mean(np.abs(tline[lined] - yt[lined]))), 2))
        ats, ou = {}, {}
        for th in (1, 2, 3, 4):
            d = pm - spread; q = lined & (np.abs(d) >= th) & (ym != spread)
            w = int(np.sum(((d > 0) == (ym > spread))[q])); ats[th] = [w, int(q.sum()) - w]
            d2 = pt - tline; q2 = lined & (np.abs(d2) >= th) & (yt != tline)
            w2 = int(np.sum(((d2 > 0) == (yt > tline))[q2])); ou[th] = [w2, int(q2.sum()) - w2]
        o['ats'], o['ou'] = ats, ou
        okw = m & ~np.isnan(yw)
        z = np.clip(pw[okw], -30, 30); y = yw[okw]
        o['logloss_model'] = round(float(np.mean(np.logaddexp(0, z) - y * z)), 4)
        zm = np.clip(ml[okw], -30, 30)
        o['logloss_market'] = round(float(np.mean(np.logaddexp(0, zm) - y * zm)), 4)
        fav = np.where(spread != 0, spread, ml)
        o['su_model'] = [int(np.sum((pw[okw] > 0) == (y > 0.5))), int(okw.sum())]
        o['su_market'] = [int(np.sum((fav[okw] > 0) == (y > 0.5))), int(okw.sum())]
        o['su_old'] = [int(np.sum((col('pred')[okw] > 0) == (y > 0.5))), int(okw.sum())]
        return o

    periods = {'2010-2015': (2010, 2015), '2016-2020': (2016, 2020), '2021-2025': (2021, 2025), '2026': (2026, 2026)}
    backtest = dict(overall=block(bt_rows),
                    by_season={int(y): block(fin & (s == y)) for y in range(2010, 2026)},
                    periods={k: block(fin & (s >= a) & (s <= b)) for k, (a, b) in periods.items()},
                    period_roles={'2010-2015': 'backtest', '2016-2020': 'validation', '2021-2025': 'locked test', '2026': 'live'})

    # genetic search summary
    ga = []
    for p in sorted(glob.glob(os.path.join(LAB, 'runs', '*.json'))):
        if 'smoke' in p: continue
        R = json.load(open(p)); bt, bv = R['best_by_train'], R['best_by_val']
        kind = 'gafs' if os.path.basename(p).startswith('gafs') else 'gp'
        row = dict(run=os.path.basename(p)[:-5], engine='feature-selection GA' if kind == 'gafs' else 'formula GP',
                   task=R['task'], null=R['null'], evals=R.get('evals'),
                   train_oof=bt['oof_acc'] if R['task'] not in ('margin', 'total') else None,
                   val_of_train_best=bt['val']['acc'], val_best=bv['val']['acc'],
                   base_val=(R.get('base_only') or {}).get('val', {}).get('acc'))
        if R['task'] in ('margin', 'total'):
            row.update(train_oof_mae=bt['oof_acc'], val_mae_train_best=bt['val']['mae'], val_mae_best=bv['val']['mae'],
                       base_val_mae=(R.get('base_only') or {}).get('val', {}).get('mae'))
        ga.append(row)

    # the plain (non-evolved) models on the same validation seasons, for comparison with each search
    vb = backtest['periods']['2016-2020']
    rate = lambda wl: round(wl[0] / max(1, wl[0] + wl[1]), 4)
    ga_ref = dict(su_ind=round(vb['su_model'][0] / vb['su_model'][1], 4), su_mkt=round(vb['su_market'][0] / vb['su_market'][1], 4),
                  ats=None, ou=None, margin=vb['mae_model'], total=vb['mae_total_model'])
    vm = fin & have & (s >= 2016) & (s <= 2020) & hl
    q = vm & (ym != spread); ga_ref['ats'] = round(float(np.mean(((pm - spread) > 0)[q] == (ym > spread)[q])), 4)
    q = vm & (yt != tline); ga_ref['ou'] = round(float(np.mean(((pt - tline) > 0)[q] == (yt > tline)[q])), 4)
    prod = W['prod']
    # spread -> moneyline log-odds slope, fitted on 2006-2015 exactly as lab/features.py does
    import importlib.util
    sys.argv = [sys.argv[0], '--offline', '--quiet']
    spec = importlib.util.spec_from_file_location('nb', os.path.join(MODEL, 'nfl-build.py'))
    nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
    imp = lambda x: 100 / (x + 100) if x > 0 else -x / (-x + 100)
    lg = lambda p: math.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4)))
    num = den = 0.0
    for g in nb.load_games():
        if 2006 <= g['season'] <= 2015 and g['home_ml'] and g['away_ml'] and g['spread_line']:
            p = imp(g['home_ml']) / (imp(g['home_ml']) + imp(g['away_ml']))
            num += g['spread_line'] ** 2; den += g['spread_line'] * lg(p)
    s2l = den / num
    out = dict(built=datetime.date.today().isoformat(), fit_seasons=prod['fit_seasons'], features=prod['features'], mu=prod['mu'], sd=prod['sd'],
               clip=prod['clip'], margin=prod['margin'], total=prod['total'], win=prod['win'],
               spread_to_logit=s2l, spec=prod['spec'],
               groups={n: group_of(n) for n in prod['features']}, residuals=resid, calib=calib, backtest=backtest, ga=ga, ga_ref=ga_ref)
    import keys                                    # key-number pricing for spreads and totals (keys.py)
    out['keys'] = keys.export(keys.load())
    json.dump(out, open(OUT, 'w'), separators=(',', ':'))
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
    print('calibration', calib)
    for k, v in backtest['periods'].items():
        print(k, {kk: v[kk] for kk in ('n', 'mae_model', 'mae_close', 'mae_total_model', 'mae_total_close', 'su_model', 'su_market', 'su_old', 'logloss_model', 'logloss_market')}, 'ATS', v['ats'], 'OU', v['ou'])
    print('spread_to_logit', round(s2l, 5)); print('ga_ref', ga_ref)
    from collections import Counter
    print(Counter(out['groups'].values()))

if __name__ == '__main__':
    main()
