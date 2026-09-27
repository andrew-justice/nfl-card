#!/usr/bin/env python3
"""
pbp.py -- condense nflverse play-by-play into one row per team per game (pure Python, streams the .csv.gz).
The same function is copied into nfl-build.py, which runs it on the current season each week.

Columns per (game_id, team), offense unless noted:
  n, epa, succ                 scrimmage plays (pass or rush, no two-point tries) with EPA
  ed_n, ed_epa, ed_succ        early downs (1st and 2nd)
  nt_n, nt_epa                 neutral game state (win probability between 10% and 90%)
  poe_n, poe                   pass rate over expected, summed over plays that have it
  d3_att, d3_conv              third-down attempts and conversions
  rz_n, rz_epa                 plays inside the opponent's 20
  st_epa                       special-teams EPA for the team (kicking and returning), net of the opponent
"""
import csv, gzip, os, sys

FIELDS = ['game_id', 'team', 'n', 'epa', 'succ', 'ed_n', 'ed_epa', 'ed_succ', 'nt_n', 'nt_epa', 'poe_n', 'poe',
          'd3_att', 'd3_conv', 'rz_n', 'rz_epa', 'st_epa']
TEAM_FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX'}

def _f(x):
    try:
        return float(x) if x not in ('', 'NA') else None
    except ValueError:
        return None

def pbp_team_rows(path):
    """Yield one dict per (game_id, team) from a play_by_play .csv.gz file."""
    agg = {}
    with gzip.open(path, 'rt', newline='') as fh:
        rd = csv.reader(fh)
        head = next(rd)
        ix = {c: i for i, c in enumerate(head)}
        need = ['game_id', 'posteam', 'defteam', 'epa', 'success', 'down', 'wp', 'pass', 'rush', 'pass_oe',
                'third_down_converted', 'third_down_failed', 'yardline_100', 'special_teams_play', 'two_point_attempt']
        I = {c: ix[c] for c in need}
        for r in rd:
            gid, pos, de = r[I['game_id']], r[I['posteam']], r[I['defteam']]
            if not gid or not pos or not de:
                continue
            pos, de = TEAM_FIX.get(pos, pos), TEAM_FIX.get(de, de)
            epa = _f(r[I['epa']])
            if epa is None:
                continue
            for t in (pos, de):
                if (gid, t) not in agg:
                    agg[(gid, t)] = dict.fromkeys(FIELDS[2:], 0.0)
            if r[I['special_teams_play']] == '1':
                agg[(gid, pos)]['st_epa'] += epa
                agg[(gid, de)]['st_epa'] -= epa
                continue
            if r[I['two_point_attempt']] == '1' or not (r[I['pass']] == '1' or r[I['rush']] == '1'):
                continue
            a = agg[(gid, pos)]
            succ = 1.0 if r[I['success']] == '1' else 0.0
            a['n'] += 1; a['epa'] += epa; a['succ'] += succ
            if r[I['down']] in ('1', '2', '1.0', '2.0'):
                a['ed_n'] += 1; a['ed_epa'] += epa; a['ed_succ'] += succ
            wp = _f(r[I['wp']])
            if wp is not None and 0.1 <= wp <= 0.9:
                a['nt_n'] += 1; a['nt_epa'] += epa
            poe = _f(r[I['pass_oe']])
            if poe is not None:
                a['poe_n'] += 1; a['poe'] += poe
            if r[I['third_down_converted']] == '1':
                a['d3_att'] += 1; a['d3_conv'] += 1
            elif r[I['third_down_failed']] == '1':
                a['d3_att'] += 1
            yl = _f(r[I['yardline_100']])
            if yl is not None and yl <= 20:
                a['rz_n'] += 1; a['rz_epa'] += epa
    for (gid, t), a in agg.items():
        yield dict(game_id=gid, team=t, **{k: round(v, 4) for k, v in a.items()})

if __name__ == '__main__':
    LAB = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(LAB, 'pbp-team.csv')
    years = range(int(sys.argv[1]) if len(sys.argv) > 1 else 1999, int(sys.argv[2]) + 1 if len(sys.argv) > 2 else 2027)
    n = 0
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for y in years:
            p = os.path.join(LAB, 'raw', 'pbp', f'pbp{y}.csv.gz')
            if not os.path.exists(p):
                continue
            for row in pbp_team_rows(p):
                w.writerow(row); n += 1
            print(y, n, flush=True)
    print('wrote', out, n, 'rows')
