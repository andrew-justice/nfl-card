#!/usr/bin/env python3
"""
features.py -- leak-free pre-kickoff feature library for every NFL game 1999-2026.

Pure Python (no numpy) so whatever the genetic search selects can be ported into nfl-build.py
and run in a-Shell.  Every feature for a game is a snapshot of state *before* that game's result
is applied; the result then updates the trackers.

Writes lab/feats.csv: meta columns, feature columns (prefixed f_), target columns (prefixed y_).
"""
import csv, math, os, sys, importlib.util

LAB = os.path.dirname(os.path.abspath(__file__))
_up = os.path.dirname(LAB)
MODEL = _up if os.path.exists(os.path.join(_up, 'nfl-build.py')) else os.path.join(_up, 'nfl-model')
sys.argv = [sys.argv[0], '--offline', '--quiet']
spec = importlib.util.spec_from_file_location('nb', os.path.join(MODEL, 'nfl-build.py'))
nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)

TEAM_FIX = nb.TEAM_FIX
COLD = {'BUF', 'GB', 'CHI', 'NE', 'CLE', 'PIT', 'NYG', 'NYJ', 'DEN', 'KC', 'CIN', 'BAL', 'PHI', 'WAS', 'SEA', 'TEN', 'MIN', 'DET', 'IND', 'STL', 'LA'}
# only teams that played outdoors in cold months count; roof filter below handles domes

def fnum(x, d=None):
    try:
        return d if x in ('', 'NA', None) else float(x)
    except ValueError:
        return d

def imp(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)

def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))

# ---------------------------------------------------------------- raw team-week stats
def load_team_week():
    """History from the package's data/team-stats.csv (1999-2025, full precision), this season from cache/."""
    tw = {}
    for path in (os.path.join(MODEL, 'data', 'team-stats.csv'), os.path.join(MODEL, 'cache', 'tw2026.csv')):
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                if not r['team'] or not r['game_id']:
                    continue
                t = TEAM_FIX.get(r['team'], r['team'])
                tw[(r['game_id'], t)] = r
    return tw

def team_obs(r, o):
    """Per-game observation for a team (r) against its opponent (o)."""
    g = lambda row, k: fnum(row.get(k), 0.0)
    att, car, sk = g(r, 'attempts'), g(r, 'carries'), g(r, 'sacks_suffered')
    oatt, ocar, osk = g(o, 'attempts'), g(o, 'carries'), g(o, 'sacks_suffered')
    plays, oplays = att + car + sk, oatt + ocar + osk
    if plays < 20 or oplays < 20:
        return None
    epa, oepa = g(r, 'passing_epa') + g(r, 'rushing_epa'), g(o, 'passing_epa') + g(o, 'rushing_epa')
    give = g(r, 'passing_interceptions') + g(r, 'fumbles_lost_total')
    ogive = g(o, 'passing_interceptions') + g(o, 'fumbles_lost_total')
    ob = dict(
        epa_off=epa / plays, epa_def=oepa / oplays, epa_pg=epa - oepa,
        pass_off=g(r, 'passing_epa') / max(1, att + sk), pass_def=g(o, 'passing_epa') / max(1, oatt + osk),
        rush_off=g(r, 'rushing_epa') / max(1, car), rush_def=g(o, 'rushing_epa') / max(1, ocar),
        fd_off=(g(r, 'passing_first_downs') + g(r, 'rushing_first_downs')) / plays,
        fd_def=(g(o, 'passing_first_downs') + g(o, 'rushing_first_downs')) / oplays,
        expl_off=(g(r, 'passing_20') + g(r, 'rushing_20')) / plays, expl_def=(g(o, 'passing_20') + g(o, 'rushing_20')) / oplays,
        sack_off=sk / max(1, att + sk), sack_def=osk / max(1, oatt + osk),
        passrate=(att + sk) / plays, plays=plays,
        give=give, take=ogive, to_margin=ogive - give,
        fum_luck=g(r, 'fumbles_lost_total') - 0.5 * g(r, 'fumbles_total'),
        pen_margin=g(o, 'penalty_yards') - g(r, 'penalty_yards'),
        ret_margin=(g(r, 'kickoff_return_yards') + g(r, 'punt_return_yards')) - (g(o, 'kickoff_return_yards') + g(o, 'punt_return_yards')),
        punt_net=(g(r, 'pt_net_yards') / g(r, 'pt_att')) if g(r, 'pt_att') > 0 else None,
        fg_luck=g(r, 'fg_made') - 0.84 * g(r, 'fg_att'),
        qbhit_def=g(r, 'def_qb_hits'), qbhit_off=g(o, 'def_qb_hits'),
    )
    c, oc = fnum(r.get('passing_cpoe')), fnum(o.get('passing_cpoe'))
    ob['cpoe_off'] = c
    ob['cpoe_def'] = oc
    return ob

# EWMA keys -> half-lives (in games)
HL = {'s': 3.0, 'm': 8.0, 'l': 20.0}
EW = {
    'margin': 'sml', 'epa_pg': 'ml', 'luck': 'ml', 'epa_off': 'ml', 'epa_def': 'ml',
    'pass_off': 'm', 'pass_def': 'm', 'rush_off': 'm', 'rush_def': 'm', 'fd_off': 'm', 'fd_def': 'm',
    'expl_off': 'm', 'expl_def': 'm', 'sack_off': 'm', 'sack_def': 'm', 'passrate': 'm', 'plays': 'm',
    'to_margin': 'ml', 'give': 'm', 'take': 'm', 'fum_luck': 'm', 'pen_margin': 'm', 'ret_margin': 'm',
    'punt_net': 'm', 'fg_luck': 'm', 'cpoe_off': 'm', 'cpoe_def': 'm', 'qbhit_def': 'm', 'qbhit_off': 'm',
    'pts_for': 'ml', 'pts_against': 'ml', 'ats': 'ml', 'ou': 'ml', 'win': 'm',
    # play-by-play (v3)
    'pb_succ_off': 'ml', 'pb_succ_def': 'ml', 'pb_ed_off': 'm', 'pb_ed_def': 'm', 'pb_nt_off': 'ml', 'pb_nt_def': 'ml',
    'pb_poe': 'm', 'pb_d3_off': 'm', 'pb_d3_def': 'm', 'pb_rz_off': 'm', 'pb_rz_def': 'm', 'pb_st': 'ml',
}
CARRY = 0.6            # fraction of last season's deviation from league mean carried into the new season
SUM_KEYS = ['pts_for_m', 'pts_for_l', 'pts_against_m', 'pts_against_l', 'epa_off_m', 'epa_off_l', 'epa_def_m', 'epa_def_l',
            'pass_off_m', 'pass_def_m', 'rush_off_m', 'rush_def_m', 'fd_off_m', 'fd_def_m', 'expl_off_m', 'expl_def_m',
            'sack_off_m', 'sack_def_m', 'passrate_m', 'plays_m', 'give_m', 'take_m', 'pen_margin_m', 'ou_m', 'ou_l',
            'cpoe_off_m', 'cpoe_def_m', 'punt_net_m',
            'pb_succ_off_m', 'pb_succ_def_m', 'pb_nt_off_m', 'pb_nt_def_m', 'pb_poe_m', 'pb_d3_off_m', 'pb_d3_def_m', 'pb_rz_off_m', 'pb_rz_def_m']

def load_pbp_team(path):
    out = {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            out[(r['game_id'], r['team'])] = {k: float(v) for k, v in r.items() if k not in ('game_id', 'team')}
    return out

def pbp_obs(p, q):
    """Per-game play-by-play observation for a team (p) against its opponent (q)."""
    ob = {}
    if p['n'] >= 20 and q['n'] >= 20:
        ob['pb_succ_off'] = p['succ'] / p['n']; ob['pb_succ_def'] = q['succ'] / q['n']
    if p['ed_n'] >= 10 and q['ed_n'] >= 10:
        ob['pb_ed_off'] = p['ed_epa'] / p['ed_n']; ob['pb_ed_def'] = q['ed_epa'] / q['ed_n']
    if p['nt_n'] >= 10 and q['nt_n'] >= 10:
        ob['pb_nt_off'] = p['nt_epa'] / p['nt_n']; ob['pb_nt_def'] = q['nt_epa'] / q['nt_n']
    if p['poe_n'] >= 10:
        ob['pb_poe'] = p['poe'] / p['poe_n']
    if p['d3_att'] >= 3: ob['pb_d3_off'] = p['d3_conv'] / p['d3_att']
    if q['d3_att'] >= 3: ob['pb_d3_def'] = q['d3_conv'] / q['d3_att']
    if p['rz_n'] >= 3: ob['pb_rz_off'] = p['rz_epa'] / p['rz_n']
    if q['rz_n'] >= 3: ob['pb_rz_def'] = q['rz_epa'] / q['rz_n']
    ob['pb_st'] = p['st_epa']
    return ob

INJ_GROUPS = ['qb', 'ol', 'skill', 'dl', 'lb', 'db']
INJ_FROM = 2013          # first season with snap counts

def injury_features(ih, ia):
    z = dict.fromkeys(INJ_GROUPS, 0.0)
    has = 1.0 if (ih is not None and ia is not None) else 0.0
    ih, ia = ih or z, ia or z
    f = {'d_inj_' + g: ih[g] - ia[g] for g in INJ_GROUPS}
    off = lambda d: d['qb'] + d['ol'] + d['skill']
    de = lambda d: d['dl'] + d['lb'] + d['db']
    f.update(d_inj_off=off(ih) - off(ia), d_inj_def=de(ih) - de(ia), s_inj_off=off(ih) + off(ia), s_inj_def=de(ih) + de(ia), has_inj=has)
    return f

def weather_features(roof, temp, wind):
    outdoors = roof in ('outdoors', 'open')
    w = (wind if wind is not None else 8.0) if outdoors else 0.0
    return dict(wx_indoor=0.0 if outdoors else 1.0, wx_wind=w, wx_wind15=max(0.0, w - 15.0),
                wx_cold=max(0.0, 40.0 - temp) if (outdoors and temp is not None) else 0.0)

class Trackers:
    def __init__(self):
        self.ew = {}          # (team, key, hl) -> value
        self.lg = {}          # key -> league mean (EWMA over team-games)
        self.season = {}      # team -> dict(w, l, gp, pd)
        self.last = {}        # team -> dict of last-game values
        self.streak = {}      # team -> (win streak signed, ats streak signed)
        self.coach = {}       # team -> (name, games)
        self.qb_starts = {}   # qb name -> career primary starts
        self.hfa = {}         # team -> EWMA of home margin residual

    def get(self, t, k, h):
        v = self.ew.get((t, k, h))
        return self.lg.get(k, 0.0) if v is None else v

    def new_season(self, teams):
        for key in list(self.ew):
            t, k, h = key
            m = self.lg.get(k, 0.0)
            self.ew[key] = m + CARRY * (self.ew[key] - m)
        for t in teams:
            self.season[t] = dict(w=0.0, l=0.0, gp=0, pd=0.0)
        for t in list(self.streak):
            self.streak[t] = (0, 0)
        for t in list(self.hfa):
            self.hfa[t] *= 0.8

    def update(self, t, obs):
        for k, hs in EW.items():
            x = obs.get(k)
            if x is None:
                continue
            lg = self.lg.get(k)
            self.lg[k] = x if lg is None else lg + 0.005 * (x - lg)
            for h in hs:
                a = 1 - 0.5 ** (1 / HL[h])
                v = self.ew.get((t, k, h))
                self.ew[(t, k, h)] = x if v is None else v + a * (x - v)

def main():
    games = nb.load_games()
    tg, qbg = nb.load_stats(2026)
    base, _ = nb.run_ratings(games, tg, qbg, nb.P)
    base_by = {f['game']['game_id']: f for f in base}

    # walk-forward blend predictions for every season 2004-2026 (fit on 2002..T-1 only)
    wf = {}
    done = [f for f in base if f['game']['result'] is not None and f['game']['season'] >= nb.FIT_FROM]
    for T in range(2004, 2027):
        tr = [f for f in done if f['game']['season'] < T]
        bs = nb.ols([nb.x_spread(f) for f in tr], [f['game']['result'] for f in tr])
        bt = nb.ols([nb.x_total(f) for f in tr], [f['game']['total'] for f in tr])
        for f in base:
            if f['game']['season'] == T:
                wf[f['game']['game_id']] = (nb.dot(bs, nb.x_spread(f)), nb.dot(bt, nb.x_total(f)))

    # ML-vs-spread mapping fitted on 2006-2015 only
    num = den = 0.0
    for g in games:
        if 2006 <= g['season'] <= 2015 and g['home_ml'] and g['away_ml'] and g['spread_line']:
            p = imp(g['home_ml']) / (imp(g['home_ml']) + imp(g['away_ml']))
            num += g['spread_line'] ** 2; den += g['spread_line'] * logit(p)
    SPREAD_TO_LOGIT = den / num
    print(f"spread->logit slope {SPREAD_TO_LOGIT:.4f} (1 pt of spread = {SPREAD_TO_LOGIT:.3f} logit)")

    tw = load_team_week()
    pbt = load_pbp_team(os.path.join(LAB, 'pbp-team.csv'))
    import glob, inj
    snaps = inj.load_snaps(sorted(glob.glob(os.path.join(LAB, 'raw', 'snap', 'snap_counts_*.csv'))))
    reports = inj.load_reports(sorted(glob.glob(os.path.join(LAB, 'raw', 'inj', 'injuries_*.csv'))))
    injmap = inj.injury_impact([g for g in games if g['season'] >= INJ_FROM], snaps, reports)
    raw_rows = {}
    with open(os.path.join(MODEL, 'cache', 'games.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            raw_rows[r['game_id']] = r

    T = Trackers()
    cur = None
    out = []
    for g in games:
        gid, h, a = g['game_id'], g['home'], g['away']
        if g['season'] != cur:
            T.new_season(set([x['home'] for x in games if x['season'] == g['season']] + [x['away'] for x in games if x['season'] == g['season']]))
            cur = g['season']
        rr = raw_rows[gid]
        f = {}
        # ---- team trackers (home minus away), plus sums for totals
        for k, hs in EW.items():
            for hh in hs:
                vh, va = T.get(h, k, hh), T.get(a, k, hh)
                f[f'd_{k}_{hh}'] = vh - va
                if f'{k}_{hh}' in SUM_KEYS:
                    f[f's_{k}_{hh}'] = vh + va
        for side, t in (('h', h), ('a', a)):
            s = T.season.get(t, dict(w=0, l=0, gp=0, pd=0))
            f[f'{side}_gp'] = s['gp']
            f[f'{side}_wpct'] = (s['w'] + 1) / (s['w'] + s['l'] + 2)
            f[f'{side}_pd'] = s['pd'] / (s['gp'] + 2)
            L = T.last.get(t, {})
            f[f'{side}_last_margin'] = L.get('margin', 0.0)
            f[f'{side}_last_ats'] = L.get('ats', 0.0)
            f[f'{side}_last_epa'] = L.get('epa_pg', 0.0)
            f[f'{side}_last_to'] = L.get('to_margin', 0.0)
            ws, ats_s = T.streak.get(t, (0, 0))
            f[f'{side}_streak'] = ws
            f[f'{side}_ats_streak'] = ats_s
            cn, cg = T.coach.get(t, ('', 0))
            coach_now = rr['home_coach'] if side == 'h' else rr['away_coach']
            tenure = cg if coach_now == cn else 0
            f[f'{side}_coach_games'] = min(tenure, 100)
            f[f'{side}_coach_new'] = 1.0 if tenure < 17 else 0.0
            qbname = base_by[gid]['qb_h' if side == 'h' else 'qb_a']
            st = T.qb_starts.get(qbname, 0)
            f[f'{side}_qb_starts'] = min(st, 150)
            f[f'{side}_qb_rookie'] = 1.0 if st < 10 else 0.0
            f[f'{side}_hfa'] = T.hfa.get(t, 0.0)
        for k in ('gp', 'wpct', 'pd', 'last_margin', 'last_ats', 'last_epa', 'last_to', 'streak', 'ats_streak',
                  'coach_games', 'coach_new', 'qb_starts', 'qb_rookie'):
            f[f'd_{k}'] = f[f'h_{k}'] - f[f'a_{k}']
        # ---- base model (existing Elo / QB / EPA ratings)
        b = base_by[gid]
        f['elo_spread'] = b['elo_spread']; f['qa_h'] = b['qa_h']; f['qa_a'] = b['qa_a']; f['qa_diff'] = b['qa_h'] - b['qa_a']
        f['epa_net'] = b['epa_net']; f['oh'] = b['oh']; f['dh'] = b['dh']; f['oa'] = b['oa']; f['da'] = b['da']
        f['pace_sum'] = b['pace']; f['lg_total'] = b['lg_total']; f['bc_early'] = b['bc_early']; f['bc_late'] = b['bc_late']
        f['qb_new_h'] = 1.0 if b['qb_new_h'] else 0.0; f['qb_new_a'] = 1.0 if b['qb_new_a'] else 0.0
        f['elo_h'] = b['elo_h'] - 1500; f['elo_a'] = b['elo_a'] - 1500
        pw = wf.get(gid)
        f['pred'] = pw[0] if pw else b['elo_spread']
        f['pred_total'] = pw[1] if pw else b['lg_total']
        f['has_pred'] = 1.0 if pw else 0.0
        # ---- schedule context
        f['week'] = g['week']; f['playoff'] = 0.0 if g['gtype'] == 'REG' else 1.0
        hour = int(g['gametime'][:2]) if g['gametime'][:2].isdigit() else 13
        f['prime'] = 1.0 if hour >= 19 else 0.0
        wd = rr.get('weekday', '')
        f['thu'] = 1.0 if wd == 'Thursday' else 0.0; f['mon'] = 1.0 if wd == 'Monday' else 0.0
        f['neutral'] = 1.0 if g['neutral'] else 0.0; f['div'] = float(g['div'])
        roof = rr.get('roof', '')
        f['dome'] = 1.0 if roof in ('dome', 'closed') else 0.0
        f['open_roof'] = 1.0 if roof == 'open' else 0.0
        month = int(g['gameday'][5:7])
        f['cold'] = 1.0 if (roof == 'outdoors' and h in COLD and month in (11, 12, 1, 2)) else 0.0
        f['grass'] = 1.0 if rr.get('surface', '').strip() in ('grass', 'dessograss') else 0.0
        f['tzdiff'] = nb.TZ.get(a, 0) - nb.TZ.get(h, 0)
        f['rest_h'] = min(g['home_rest'], 14); f['rest_a'] = min(g['away_rest'], 14)
        f['rest_diff'] = f['rest_h'] - f['rest_a']
        f['short_h'] = 1.0 if g['home_rest'] <= 5 else 0.0; f['short_a'] = 1.0 if g['away_rest'] <= 5 else 0.0
        f['bye_h'] = 1.0 if (g['home_rest'] >= 13 and g['week'] > 1) else 0.0
        f['bye_a'] = 1.0 if (g['away_rest'] >= 13 and g['week'] > 1) else 0.0
        # ---- market (closing numbers as published pre-kickoff)
        sp, tl = g['spread_line'], g['total_line']
        f['has_line'] = 1.0 if sp is not None else 0.0
        f['spread'] = sp if sp is not None else f['pred']
        f['total_line'] = tl if tl is not None else f['pred_total']
        if g['home_ml'] and g['away_ml']:
            p = imp(g['home_ml']) / (imp(g['home_ml']) + imp(g['away_ml']))
            f['ml_logit'] = logit(p); f['ml_gap'] = logit(p) - SPREAD_TO_LOGIT * f['spread']
        else:
            f['ml_logit'] = SPREAD_TO_LOGIT * f['spread']; f['ml_gap'] = 0.0
        hso, aso = fnum(rr.get('home_spread_odds')), fnum(rr.get('away_spread_odds'))
        f['juice_s'] = (imp(hso) - imp(aso)) if (hso and aso) else 0.0
        oo, uo = fnum(rr.get('over_odds')), fnum(rr.get('under_odds'))
        f['juice_t'] = (imp(oo) - imp(uo)) if (oo and uo) else 0.0
        f['imp_h'] = (f['total_line'] + f['spread']) / 2; f['imp_a'] = (f['total_line'] - f['spread']) / 2
        f['model_gap'] = f['pred'] - f['spread']; f['model_gap_t'] = f['pred_total'] - f['total_line']
        # ---- v3: injuries (final report x recent snap share) and game-time weather
        f.update(injury_features(injmap.get((gid, h)), injmap.get((gid, a))))
        f.update(weather_features(rr.get('roof', ''), fnum(rr.get('temp')), fnum(rr.get('wind'))))

        # ---- targets
        y = {}
        if g['result'] is not None:
            m = g['result']; tot = g['total']
            y['y_margin'] = m; y['y_total'] = tot
            y['y_win'] = 1.0 if m > 0 else (0.0 if m < 0 else '')
            if sp is not None:
                y['y_ats'] = m - sp; y['y_cover'] = 1.0 if m > sp else (0.0 if m < sp else '')
            if tl is not None:
                y['y_ou'] = tot - tl; y['y_over'] = 1.0 if tot > tl else (0.0 if tot < tl else '')
        out.append(dict(game_id=gid, season=g['season'], week=g['week'], gtype=g['gtype'], home=h, away=a,
                        final=1 if g['result'] is not None else 0, **{'f_' + k: v for k, v in f.items()}, **y))

        # ---- apply result
        if g['result'] is None:
            continue
        rh, ra = tw.get((gid, h)), tw.get((gid, a))
        ph, pa = pbt.get((gid, h)), pbt.get((gid, a))
        m = g['result']
        for t, r, o, sgn, pp, pq in ((h, rh, ra, 1, ph, pa), (a, ra, rh, -1, pa, ph)):
            ob = team_obs(r, o) if (r and o) else None
            ob = ob or {}
            if pp and pq:
                ob.update(pbp_obs(pp, pq))
            mm = sgn * m
            ob['margin'] = mm
            ob['pts_for'] = g['home_score'] if sgn == 1 else g['away_score']
            ob['pts_against'] = g['away_score'] if sgn == 1 else g['home_score']
            if 'epa_pg' in ob:
                ob['luck'] = mm - ob['epa_pg']
            if sp is not None:
                ob['ats'] = sgn * (m - sp)
            if tl is not None:
                ob['ou'] = g['total'] - tl
            ob['win'] = 1.0 if mm > 0 else (0.0 if mm < 0 else 0.5)
            T.update(t, ob)
            s = T.season.setdefault(t, dict(w=0.0, l=0.0, gp=0, pd=0.0))
            s['gp'] += 1; s['pd'] += mm
            if mm > 0: s['w'] += 1
            elif mm < 0: s['l'] += 1
            else: s['w'] += 0.5; s['l'] += 0.5
            T.last[t] = dict(margin=mm, ats=ob.get('ats', 0.0), epa_pg=ob.get('epa_pg', 0.0), to_margin=ob.get('to_margin', 0.0))
            ws, a_s = T.streak.get(t, (0, 0))
            ws = (ws + 1 if ws > 0 else 1) if mm > 0 else ((ws - 1 if ws < 0 else -1) if mm < 0 else 0)
            if 'ats' in ob:
                a_s = (a_s + 1 if a_s > 0 else 1) if ob['ats'] > 0 else ((a_s - 1 if a_s < 0 else -1) if ob['ats'] < 0 else 0)
            T.streak[t] = (ws, a_s)
            coach_now = raw_rows[gid]['home_coach'] if sgn == 1 else raw_rows[gid]['away_coach']
            cn, cg = T.coach.get(t, ('', 0))
            T.coach[t] = (coach_now, cg + 1 if coach_now == cn else 1)
            q = qbg.get((gid, t))
            if q:
                T.qb_starts[q['name']] = T.qb_starts.get(q['name'], 0) + 1
        if not g['neutral']:
            exp = b['elo_spread']            # home margin the ratings expected (includes league HFA)
            T.hfa[h] = T.hfa.get(h, 0.0) + 0.05 * ((m - exp) - T.hfa.get(h, 0.0))

    cols = list(out[-1].keys())
    for r in out:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(os.path.join(LAB, 'feats.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, '') for c in cols})
    nf = sum(1 for c in cols if c.startswith('f_'))
    print(f"wrote feats.csv: {len(out)} games, {nf} features")

if __name__ == '__main__':
    main()
