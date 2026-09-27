#!/usr/bin/env python3
"""
inj.py -- injury impact per team per game, from final injury reports and snap counts (pure Python).
The same code is copied into nfl-build.py for the current season.

A player's importance is his average snap share over his team's previous four games (offense share for
offensive positions, defense share for defensive ones; zero for games he missed).  A player's chance of
missing the game comes from his final report status.  Impact for a position group = sum of
chance-of-missing x importance over the group's listed players.
"""
import bisect, csv, os, re

TEAM_FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX'}
P_MISS = {'Out': 1.0, 'Doubtful': 0.9, 'Questionable': 0.25}
GROUP = {'QB': 'qb', 'T': 'ol', 'OT': 'ol', 'G': 'ol', 'OG': 'ol', 'C': 'ol', 'OL': 'ol',
         'WR': 'skill', 'TE': 'skill', 'RB': 'skill', 'FB': 'skill', 'HB': 'skill',
         'DE': 'dl', 'DT': 'dl', 'NT': 'dl', 'DL': 'dl', 'EDGE': 'dl',
         'LB': 'lb', 'ILB': 'lb', 'OLB': 'lb', 'MLB': 'lb',
         'CB': 'db', 'S': 'db', 'SS': 'db', 'FS': 'db', 'DB': 'db', 'SAF': 'db'}
OFFENSE = {'qb', 'ol', 'skill'}
GROUPS = ['qb', 'ol', 'skill', 'dl', 'lb', 'db']
SUFFIX = re.compile(r'\b(jr|sr|ii|iii|iv|v)\b')

def norm(name):
    n = name.lower().replace('.', '').replace("'", '').replace('-', ' ').replace(',', ' ')
    return ' '.join(SUFFIX.sub('', n).split())

def load_snaps(paths):
    """team -> chronological list of (game_id, {name: (off_share, def_share)})"""
    games = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                t = TEAM_FIX.get(r['team'], r['team'])
                key = (int(r['season']), int(r['week']), r['game_id'])
                d = games.setdefault(t, {}).setdefault(key, {})
                try:
                    d[norm(r['player'])] = (float(r['offense_pct'] or 0), float(r['defense_pct'] or 0))
                except ValueError:
                    pass
    return {t: [((k[0], k[1]), k[2], v) for k, v in sorted(g.items())] for t, g in games.items()}

def load_reports(paths):
    """(season, team, week) -> list of (name, group, p_miss)"""
    rep = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                pm = P_MISS.get(r.get('report_status', ''))
                grp = GROUP.get(r.get('position', ''))
                if not pm or not grp:
                    continue
                t = TEAM_FIX.get(r['team'], r['team'])
                rep.setdefault((int(r['season']), t, int(r['week'])), []).append((norm(r['full_name']), grp, pm))
    return rep

def injury_impact(games, snaps, reports):
    """games: list of dicts with game_id, season, week, home, away.
    Returns (game_id, team) -> dict of impact by group, for teams with a report that week."""
    keys = {t: [k for k, _, _ in lst] for t, lst in snaps.items()}
    out = {}
    for g in games:
        for t in (g['home'], g['away']):
            rep = reports.get((g['season'], t, g['week']))
            if rep is None:
                continue
            lst = snaps.get(t, [])
            i = bisect.bisect_left(keys.get(t, []), (g['season'], g['week']))   # only games before this one
            prev = lst[max(0, i - 4):i]
            imp = {}
            if prev:
                for _, _, d in prev:
                    for name in d:
                        imp.setdefault(name, None)
                for name in imp:
                    o = sum(d.get(name, (0.0, 0.0))[0] for _, _, d in prev) / len(prev)
                    df = sum(d.get(name, (0.0, 0.0))[1] for _, _, d in prev) / len(prev)
                    imp[name] = (o, df)
            last = {}
            for name in imp:
                last.setdefault(name.split(' ')[-1], []).append(name)
            f = dict.fromkeys(GROUPS, 0.0)
            for name, grp, pm in rep:
                if name not in imp:                      # nickname or spelling variant: unique last-name match on the roster
                    cand = last.get(name.split(' ')[-1], [])
                    if len(cand) == 1:
                        name = cand[0]
                o, df = imp.get(name, (0.0, 0.0))
                f[grp] += pm * (o if grp in OFFENSE else df)
            out[(g['game_id'], t)] = {k: round(v, 4) for k, v in f.items()}
    return out
