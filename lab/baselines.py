#!/usr/bin/env python3
"""Reference points for the genetic search: market, current model, and standard ML fitted on 2002-2015, scored on 2016-2020."""
import numpy as np, sys, os, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp

D = gp.load()
names = D['names']; col = lambda n: D['X'][:, names.index(n)]
s = D['season']; fin = D['final'] == 1
val = fin & (s >= 2016) & (s <= 2020)
tr = fin & (s >= 2002) & (s <= 2015)

def acc(pred_home, y, m):
    m = m & ~np.isnan(y)
    return round(float(np.mean((pred_home[m] > 0) == (y[m] > 0.5))), 4), int(m.sum())

yw, yc, yo = D['Y']['y_win'], D['Y']['y_cover'], D['Y']['y_over']
spread, ml = col('spread'), col('ml_logit')
fav = np.where(spread != 0, spread, ml)
print('SU  market favorite      val', acc(fav, yw, val), ' train', acc(fav, yw, tr))
print('SU  current model (pred) val', acc(col('pred'), yw, val), ' train', acc(col('pred'), yw, tr))
print('SU  Elo only             val', acc(col('elo_spread'), yw, val))
print('ATS always home          val', acc(np.ones(len(s)), yc, val), ' train', acc(np.ones(len(s)), yc, tr))
print('ATS model side           val', acc(col('model_gap'), yc, val), ' train', acc(col('model_gap'), yc, tr))
print('O/U always over          val', acc(np.ones(len(s)), yo, val), ' train', acc(np.ones(len(s)), yo, tr))
print('O/U model side           val', acc(col('model_gap_t'), yo, val), ' train', acc(col('model_gap_t'), yo, tr))

for task in ('su_ind', 'su_mkt', 'ats', 'ou'):
    P = gp.prepare(D, task)
    Z, y = P['Z'], P['y']
    trm, vam = P['tr'], P['va']
    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1):
        oof = np.zeros(trm.sum()); Zt, yt, ft = Z[trm], y[trm], P['fold'][trm]
        for f in range(4):
            m = ft == f
            lr = LogisticRegression(C=C, max_iter=2000).fit(Zt[~m], yt[~m]); oof[m] = lr.decision_function(Zt[m])
        a = float(np.mean((oof > 0) == (yt > 0.5)))
        if best is None or a > best[0]: best = (a, C)
    lr = LogisticRegression(C=best[1], max_iter=2000).fit(Z[trm], y[trm])
    va_acc = float(np.mean((lr.decision_function(Z[vam]) > 0) == (y[vam] > 0.5)))
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, l2_regularization=1.0, random_state=0).fit(Z[trm], y[trm])
    hv = float(np.mean((hgb.predict_proba(Z[vam])[:, 1] > 0.5) == (y[vam] > 0.5)))
    print(f'{task:7s} logistic (C={best[1]}) OOF-train {best[0]:.4f} val {va_acc:.4f} | gradient boosting val {hv:.4f} | n_val {vam.sum()}')
