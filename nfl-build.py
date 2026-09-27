#!/usr/bin/env python3
"""
nfl-build.py  --  pandas-free NFL prediction model + dashboard builder.

Weekly use:   python3 nfl-build.py            (downloads this season's data, rebuilds nfl-dashboard.html)
Offline:      python3 nfl-build.py --offline  (uses cache/ as-is)
Re-tune:      python3 nfl-build.py --tune     (slow; coordinate-descent over model constants)

Files:
  data/qb-games.csv, data/team-games.csv   historical caches shipped with the tool (1999-2025)
  data/team-stats.csv                       richer team-game stats for the feature library (1999-2025)
  data/model.json                           fitted model: coefficients, residuals, calibration, backtest (from lab/)
  cache/                                    downloaded current files (games.csv + this season's stats)
  nfl-picks.csv                              auto-maintained pick log (graded on later runs)
  dashboard-template.html                    UI; the script injects JSON at /*__DATA__*/
  nfl-dashboard.html                         output
"""
import bisect, csv, gzip, json, math, os, re, sys, time, random, datetime, urllib.request, urllib.error

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, CACHE = os.path.join(BASE, 'data'), os.path.join(BASE, 'cache')
os.makedirs(CACHE, exist_ok=True)
ARGS = set(a for a in sys.argv[1:])
QUIET = '--quiet' in ARGS

URL_GAMES = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
URL_PW = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{s}.csv"
URL_TW = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{s}.csv"
URL_PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.csv.gz"
URL_INJ = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{s}.csv"
URL_SNAP = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{s}.csv"
URL_WX = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m"
          "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%2FNew_York&forecast_days=16")
SHIPPED_THROUGH = 2025    # data/*.csv hold history through this season; later seasons are downloaded and cached

# ---------------- tuned constants (see --tune) ----------------
P = dict(K=20.0, HFA=30.0, REST=0.0, REGRESS=0.333, MEAN=1505.0,
         QB_MULT=4.5, QB_ALPHA=0.06, QB_PRIOR_GAP=30.0, QB_OFF=0.15,
         EPA_HL=12.0, EPA_CARRY=0.3, MOV_W=1.0, PACE_HL=8.0, LG_A=0.02)
FIT_FROM, WF_FROM, RESID_FROM = 2002, 2010, 2014
EDGE_MIN = 0.01           # log a pick when market-calibrated edge >= 1% at -110
KELLY_FRAC, KELLY_CAP = 0.25, 0.02

TEAM_FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
TZ = {'SEA': -3, 'SF': -3, 'LA': -3, 'LAC': -3, 'LV': -3, 'DEN': -2, 'ARI': -2,
      'CHI': -1, 'GB': -1, 'MIN': -1, 'DAL': -1, 'HOU': -1, 'KC': -1, 'NO': -1, 'TEN': -1}
DIV = {'BUF': 'AFC East', 'MIA': 'AFC East', 'NE': 'AFC East', 'NYJ': 'AFC East',
       'BAL': 'AFC North', 'CIN': 'AFC North', 'CLE': 'AFC North', 'PIT': 'AFC North',
       'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South', 'TEN': 'AFC South',
       'DEN': 'AFC West', 'KC': 'AFC West', 'LV': 'AFC West', 'LAC': 'AFC West',
       'DAL': 'NFC East', 'NYG': 'NFC East', 'PHI': 'NFC East', 'WAS': 'NFC East',
       'CHI': 'NFC North', 'DET': 'NFC North', 'GB': 'NFC North', 'MIN': 'NFC North',
       'ATL': 'NFC South', 'CAR': 'NFC South', 'NO': 'NFC South', 'TB': 'NFC South',
       'ARI': 'NFC West', 'LA': 'NFC West', 'SEA': 'NFC West', 'SF': 'NFC West'}

def log(*a):
    if not QUIET: print(*a, flush=True)

def fnum(x, default=None):
    try:
        if x in ('', 'NA', None): return default
        return float(x)
    except ValueError:
        return default

# ---------------- data loading ----------------
def fetch(url, path):
    if '--offline' in ARGS and os.path.exists(path): return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'nfl-build/1.0'})
        with urllib.request.urlopen(req, timeout=90) as r, open(path, 'wb') as f:
            f.write(r.read())
        log(f"  fetched {os.path.basename(path)}")
    except Exception as e:
        if os.path.exists(path):
            log(f"  WARNING: download failed ({e}); using cached {os.path.basename(path)}")
        else:
            sys.exit(f"Download failed and no cache: {url}\n  In a-Shell try:  curl -L -o {path} \"{url}\"\n  then rerun with --offline")

def fetch_optional(url, path, once=False):
    """Like fetch, but a failure is a warning, not a stop: these inputs improve the numbers but are not essential."""
    if once and os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    if '--offline' in ARGS:
        return os.path.exists(path)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'nfl-build/1.0'})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + '.part', 'wb') as f:
            f.write(r.read())
        os.replace(path + '.part', path)
        log(f"  fetched {os.path.basename(path)}")
        return True
    except Exception as e:
        log(f"  WARNING: could not download {os.path.basename(path)} ({e}){'; using cached copy' if os.path.exists(path) else ''}")
        return os.path.exists(path)

def fetch_once(url, path):
    """Completed seasons never change: download once, then reuse the cached copy."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    fetch(url, path)

def load_games():
    path = os.path.join(CACHE, 'games.csv')
    fetch(URL_GAMES, path)
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            g = dict(game_id=r['game_id'], season=int(r['season']), week=int(r['week']), gtype=r['game_type'],
                     gameday=r['gameday'], gametime=r['gametime'] or '13:00',
                     home=TEAM_FIX.get(r['home_team'], r['home_team']), away=TEAM_FIX.get(r['away_team'], r['away_team']),
                     home_score=fnum(r['home_score']), away_score=fnum(r['away_score']), result=fnum(r['result']),
                     total=fnum(r['total']), neutral=(r['location'] == 'Neutral'),
                     home_rest=fnum(r['home_rest'], 7), away_rest=fnum(r['away_rest'], 7),
                     spread_line=fnum(r['spread_line']), total_line=fnum(r['total_line']),
                     home_ml=fnum(r['home_moneyline']), away_ml=fnum(r['away_moneyline']),
                     div=int(fnum(r['div_game'], 0)), roof=r['roof'], home_qb=r['home_qb_name'], away_qb=r['away_qb_name'],
                     weekday=r.get('weekday', ''), surface=r.get('surface', ''), home_coach=r.get('home_coach', ''), away_coach=r.get('away_coach', ''),
                     home_spread_odds=fnum(r.get('home_spread_odds')), away_spread_odds=fnum(r.get('away_spread_odds')),
                     over_odds=fnum(r.get('over_odds')), under_odds=fnum(r.get('under_odds')),
                     temp=fnum(r.get('temp')), wind=fnum(r.get('wind')))
            rows.append(g)
    rows.sort(key=lambda g: (g['season'], g['gameday'], g['gametime'], g['game_id']))
    return rows

def load_stats(current_season):
    """team-game and QB-game dicts keyed by (game_id, team). Historical cache + fresh current season."""
    tg, qb = {}, {}
    def read_team(path, only_season=None):
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                s = int(r['season'])
                if only_season is not None and s != only_season: continue
                if only_season is None and s >= current_season: continue
                t = TEAM_FIX.get(r['team'], r['team'])
                tg[(r['game_id'], t)] = dict(pepa=fnum(r['pepa'], 0), repa=fnum(r['repa'], 0), plays=fnum(r['plays'], 0),
                                            int=fnum(r['int'], 0), fuml=fnum(r['fuml'], 0), defint=fnum(r['defint'], 0))
    def read_qb(path, only_season=None):
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                s = int(r['season'])
                if only_season is not None and s != only_season: continue
                if only_season is None and s >= current_season: continue
                t = TEAM_FIX.get(r['team'], r['team'])
                k = (r['game_id'], t)
                cur = qb.get(k)
                if cur is None or int(r['att']) > cur['att']:
                    qb[k] = dict(name=r['name'], att=int(r['att']), cmp=int(r['cmp']), pyds=int(r['pyds']), ptd=int(r['ptd']),
                                 int=int(r['int']), sk=int(r['sk']), car=int(r['car']), ryds=int(r['ryds']), rtd=int(r['rtd']))
    read_team(os.path.join(DATA, 'team-games.csv'))
    read_qb(os.path.join(DATA, 'qb-games.csv'))
    # seasons after the shipped history: fetch raw weekly files and condense on the fly
    for season in range(SHIPPED_THROUGH + 1, current_season + 1):
        read_raw_season(season, current_season, tg, qb)
    return tg, qb

def read_raw_season(season, current_season, tg, qb):
    pw, tw = os.path.join(CACHE, f'pw{season}.csv'), os.path.join(CACHE, f'tw{season}.csv')
    get = fetch if season == current_season else fetch_once
    get(URL_PW.format(s=season), pw); get(URL_TW.format(s=season), tw)
    with open(tw, newline='') as fh:
        for r in csv.DictReader(fh):
            if not r['team'] or not r['game_id']: continue
            att, car, sk = fnum(r['attempts'], 0), fnum(r['carries'], 0), fnum(r['sacks_suffered'], 0)
            fuml = fnum(r['sack_fumbles_lost'], 0) + fnum(r['rushing_fumbles_lost'], 0) + fnum(r['receiving_fumbles_lost'], 0)
            tg[(r['game_id'], r['team'])] = dict(pepa=fnum(r['passing_epa'], 0), repa=fnum(r['rushing_epa'], 0), plays=att+car+sk,
                                                int=fnum(r['passing_interceptions'], 0), fuml=fuml, defint=fnum(r['def_interceptions'], 0))
    with open(pw, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['position'] != 'QB' or not r['game_id']: continue
            att = int(fnum(r['attempts'], 0)); k = (r['game_id'], r['team'])
            if att + int(fnum(r['carries'], 0)) == 0: continue
            cur = qb.get(k)
            if cur is None or att > cur['att']:
                qb[k] = dict(name=r['player_display_name'], att=att, cmp=int(fnum(r['completions'], 0)), pyds=int(fnum(r['passing_yards'], 0)),
                             ptd=int(fnum(r['passing_tds'], 0)), int=int(fnum(r['passing_interceptions'], 0)), sk=int(fnum(r['sacks_suffered'], 0)),
                             car=int(fnum(r['carries'], 0)), ryds=int(fnum(r['rushing_yards'], 0)), rtd=int(fnum(r['rushing_tds'], 0)))
    return tg, qb

def qb_value(q):
    return (-2.2*q['att'] + 3.7*q['cmp'] + q['pyds']/5 + 11.3*q['ptd'] - 14.1*q['int'] - 8*q['sk']
            - 1.1*q['car'] + 0.6*q['ryds'] + 15.9*q['rtd'])


# ---------------- rich pre-kickoff features (ported from lab/features.py; must stay identical) ----------------
COLD = {'BUF', 'GB', 'CHI', 'NE', 'CLE', 'PIT', 'NYG', 'NYJ', 'DEN', 'KC', 'CIN', 'BAL', 'PHI', 'WAS', 'SEA', 'TEN', 'MIN', 'DET', 'IND', 'STL', 'LA'}
FHL = {'s': 3.0, 'm': 8.0, 'l': 20.0}
EW = {
    'margin': 'sml', 'epa_pg': 'ml', 'luck': 'ml', 'epa_off': 'ml', 'epa_def': 'ml',
    'pass_off': 'm', 'pass_def': 'm', 'rush_off': 'm', 'rush_def': 'm', 'fd_off': 'm', 'fd_def': 'm',
    'expl_off': 'm', 'expl_def': 'm', 'sack_off': 'm', 'sack_def': 'm', 'passrate': 'm', 'plays': 'm',
    'to_margin': 'ml', 'give': 'm', 'take': 'm', 'fum_luck': 'm', 'pen_margin': 'm', 'ret_margin': 'm',
    'punt_net': 'm', 'fg_luck': 'm', 'cpoe_off': 'm', 'cpoe_def': 'm', 'qbhit_def': 'm', 'qbhit_off': 'm',
    'pts_for': 'ml', 'pts_against': 'ml', 'ats': 'ml', 'ou': 'ml', 'win': 'm',
    'pb_succ_off': 'ml', 'pb_succ_def': 'ml', 'pb_ed_off': 'm', 'pb_ed_def': 'm', 'pb_nt_off': 'ml', 'pb_nt_def': 'ml',
    'pb_poe': 'm', 'pb_d3_off': 'm', 'pb_d3_def': 'm', 'pb_rz_off': 'm', 'pb_rz_def': 'm', 'pb_st': 'ml',
}
CARRY = 0.6
SUM_KEYS = ['pts_for_m', 'pts_for_l', 'pts_against_m', 'pts_against_l', 'epa_off_m', 'epa_off_l', 'epa_def_m', 'epa_def_l',
            'pass_off_m', 'pass_def_m', 'rush_off_m', 'rush_def_m', 'fd_off_m', 'fd_def_m', 'expl_off_m', 'expl_def_m',
            'sack_off_m', 'sack_def_m', 'passrate_m', 'plays_m', 'give_m', 'take_m', 'pen_margin_m', 'ou_m', 'ou_l',
            'cpoe_off_m', 'cpoe_def_m', 'punt_net_m',
            'pb_succ_off_m', 'pb_succ_def_m', 'pb_nt_off_m', 'pb_nt_def_m', 'pb_poe_m', 'pb_d3_off_m', 'pb_d3_def_m', 'pb_rz_off_m', 'pb_rz_def_m']

# ---------------- play-by-play (ported from lab/pbp.py and lab/features.py) ----------------
PBP_FIELDS = ['game_id', 'team', 'n', 'epa', 'succ', 'ed_n', 'ed_epa', 'ed_succ', 'nt_n', 'nt_epa', 'poe_n', 'poe',
              'd3_att', 'd3_conv', 'rz_n', 'rz_epa', 'st_epa']
PBP_FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX'}

def pbp_team_rows(path):
    """Condense a play_by_play .csv.gz into one row per (game_id, team)."""
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
            pos, de = PBP_FIX.get(pos, pos), PBP_FIX.get(de, de)
            epa = fnum(r[I['epa']])
            if epa is None:
                continue
            for t in (pos, de):
                if (gid, t) not in agg:
                    agg[(gid, t)] = dict.fromkeys(PBP_FIELDS[2:], 0.0)
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
            wp = fnum(r[I['wp']])
            if wp is not None and 0.1 <= wp <= 0.9:
                a['nt_n'] += 1; a['nt_epa'] += epa
            poe = fnum(r[I['pass_oe']])
            if poe is not None:
                a['poe_n'] += 1; a['poe'] += poe
            if r[I['third_down_converted']] == '1':
                a['d3_att'] += 1; a['d3_conv'] += 1
            elif r[I['third_down_failed']] == '1':
                a['d3_att'] += 1
            yl = fnum(r[I['yardline_100']])
            if yl is not None and yl <= 20:
                a['rz_n'] += 1; a['rz_epa'] += epa
    return {k: {f: round(v, 4) for f, v in a.items()} for k, a in agg.items()}

def load_pbp_team(current_season):
    """(game_id, team) -> per-game play-by-play aggregates: shipped history plus every later season."""
    out = {}
    with open(os.path.join(DATA, 'pbp-team.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            if int(r['game_id'][:4]) > SHIPPED_THROUGH:
                continue
            out[(r['game_id'], r['team'])] = {k: float(v) for k, v in r.items() if k not in ('game_id', 'team')}
    for season in range(SHIPPED_THROUGH + 1, current_season + 1):
        path = os.path.join(CACHE, f'pbp{season}.csv.gz')
        if fetch_optional(URL_PBP.format(s=season), path, once=season != current_season):
            out.update(pbp_team_rows(path))
        else:
            log(f"  WARNING: no play-by-play for {season}; those ratings stay at their last values")
    return out

def pbp_obs(p, q):
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

# ---------------- injuries (ported from lab/inj.py and lab/features.py) ----------------
INJ_P_MISS = {'Out': 1.0, 'Doubtful': 0.9, 'Questionable': 0.25}
INJ_GROUP = {'QB': 'qb', 'T': 'ol', 'OT': 'ol', 'G': 'ol', 'OG': 'ol', 'C': 'ol', 'OL': 'ol',
             'WR': 'skill', 'TE': 'skill', 'RB': 'skill', 'FB': 'skill', 'HB': 'skill',
             'DE': 'dl', 'DT': 'dl', 'NT': 'dl', 'DL': 'dl', 'EDGE': 'dl',
             'LB': 'lb', 'ILB': 'lb', 'OLB': 'lb', 'MLB': 'lb',
             'CB': 'db', 'S': 'db', 'SS': 'db', 'FS': 'db', 'DB': 'db', 'SAF': 'db'}
INJ_OFFENSE = {'qb', 'ol', 'skill'}
INJ_GROUPS = ['qb', 'ol', 'skill', 'dl', 'lb', 'db']
INJ_FROM = 2013
NAME_SUFFIX = re.compile(r'\b(jr|sr|ii|iii|iv|v)\b')

def norm_name(name):
    n = name.lower().replace('.', '').replace("'", '').replace('-', ' ').replace(',', ' ')
    return ' '.join(NAME_SUFFIX.sub('', n).split())

def load_snaps(paths):
    games = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                t = PBP_FIX.get(r['team'], r['team'])
                key = (int(r['season']), int(r['week']), r['game_id'])
                d = games.setdefault(t, {}).setdefault(key, {})
                try:
                    d[norm_name(r['player'])] = (float(r['offense_pct'] or 0), float(r['defense_pct'] or 0))
                except ValueError:
                    pass
    return {t: [((k[0], k[1]), k[2], v) for k, v in sorted(g.items())] for t, g in games.items()}

def load_reports(paths):
    rep = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                pm = INJ_P_MISS.get(r.get('report_status', ''))
                grp = INJ_GROUP.get(r.get('position', ''))
                if not pm or not grp:
                    continue
                t = PBP_FIX.get(r['team'], r['team'])
                rep.setdefault((int(r['season']), t, int(r['week'])), []).append(
                    (norm_name(r['full_name']), grp, pm, r['full_name'], r.get('position', ''), r.get('report_status', '')))
    return rep

def injury_impact(games, snaps, reports):
    """(game_id, team) -> (impact by position group, list of notable players out) for teams with a report that week."""
    keys = {t: [k for k, _, _ in lst] for t, lst in snaps.items()}
    out = {}
    for g in games:
        for t in (g['home'], g['away']):
            rep = reports.get((g['season'], t, g['week']))
            if rep is None:
                continue
            lst = snaps.get(t, [])
            i = bisect.bisect_left(keys.get(t, []), (g['season'], g['week']))
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
            f = dict.fromkeys(INJ_GROUPS, 0.0); notable = []
            for name, grp, pm, full, pos, status in rep:
                if name not in imp:
                    cand = last.get(name.split(' ')[-1], [])
                    if len(cand) == 1:
                        name = cand[0]
                o, df = imp.get(name, (0.0, 0.0))
                share = o if grp in INJ_OFFENSE else df
                f[grp] += pm * share
                if share >= 0.4:
                    notable.append((round(pm * share, 3), full, pos, status, round(share, 2)))
            notable.sort(reverse=True)
            out[(g['game_id'], t)] = ({k: round(v, 4) for k, v in f.items()}, notable)
    return out

def injury_features(ih, ia):
    z = dict.fromkeys(INJ_GROUPS, 0.0)
    has = 1.0 if (ih is not None and ia is not None) else 0.0
    ih, ia = ih or z, ia or z
    f = {'d_inj_' + g: ih[g] - ia[g] for g in INJ_GROUPS}
    off = lambda d: d['qb'] + d['ol'] + d['skill']
    de = lambda d: d['dl'] + d['lb'] + d['db']
    f.update(d_inj_off=off(ih) - off(ia), d_inj_def=de(ih) - de(ia), s_inj_off=off(ih) + off(ia), s_inj_def=de(ih) + de(ia), has_inj=has)
    return f

# ---------------- weather ----------------
def weather_features(roof, temp, wind):
    outdoors = roof in ('outdoors', 'open')
    w = (wind if wind is not None else 8.0) if outdoors else 0.0
    return dict(wx_indoor=0.0 if outdoors else 1.0, wx_wind=w, wx_wind15=max(0.0, w - 15.0),
                wx_cold=max(0.0, 40.0 - temp) if (outdoors and temp is not None) else 0.0)

STADIUM = {'ARI': (33.5276, -112.2626), 'ATL': (33.7554, -84.4008), 'BAL': (39.2780, -76.6227), 'BUF': (42.7738, -78.7870),
           'CAR': (35.2258, -80.8528), 'CHI': (41.8623, -87.6167), 'CIN': (39.0955, -84.5160), 'CLE': (41.5061, -81.6995),
           'DAL': (32.7473, -97.0945), 'DEN': (39.7439, -105.0201), 'DET': (42.3400, -83.0456), 'GB': (44.5013, -88.0622),
           'HOU': (29.6847, -95.4107), 'IND': (39.7601, -86.1639), 'JAX': (30.3239, -81.6373), 'KC': (39.0489, -94.4839),
           'LV': (36.0909, -115.1833), 'LAC': (33.9535, -118.3392), 'LA': (33.9535, -118.3392), 'MIA': (25.9580, -80.2389),
           'MIN': (44.9737, -93.2581), 'NE': (42.0909, -71.2643), 'NO': (29.9511, -90.0812), 'NYG': (40.8135, -74.0745),
           'NYJ': (40.8135, -74.0745), 'PHI': (39.9008, -75.1675), 'PIT': (40.4468, -80.0158), 'SF': (37.4030, -121.9700),
           'SEA': (47.5952, -122.3316), 'TB': (27.9759, -82.5033), 'TEN': (36.1665, -86.7713), 'WAS': (38.9076, -76.8645)}

def forecast_weather(upcoming):
    """Kickoff-hour temperature and wind for upcoming outdoor games within 16 days, from Open-Meteo (free, no key).
    Returns game_id -> (temp F, wind mph).  Any failure leaves a game out, and the model then uses average wind."""
    out = {}
    if '--offline' in ARGS or '--no-weather' in ARGS:
        return out
    today = now_et().date()
    want = [g for g in upcoming if g['roof'] in ('outdoors', 'open') and not g['neutral'] and g['home'] in STADIUM
            and 0 <= (datetime.date.fromisoformat(g['gameday']) - today).days <= 15]
    cache = {}; down = None
    for g in want:
        loc = STADIUM[g['home']]
        if loc not in cache and down:
            cache[loc] = {}
        if loc not in cache:
            try:
                req = urllib.request.Request(URL_WX.format(lat=loc[0], lon=loc[1]), headers={'User-Agent': 'nfl-build/1.0'})
                with urllib.request.urlopen(req, timeout=30) as r:
                    j = json.loads(r.read().decode('utf-8'))
                if isinstance(j, list):
                    j = j[0]
                hr = j['hourly']
                cache[loc] = {t: (tf, w) for t, tf, w in zip(hr['time'], hr['temperature_2m'], hr['wind_speed_10m'])}
            except Exception as e:
                down = str(e); cache[loc] = {}
        key = f"{g['gameday']}T{g['gametime'][:2]}:00"
        v = cache[loc].get(key)
        if v and v[0] is not None and v[1] is not None:
            out[g['game_id']] = (float(v[0]), float(v[1]))
    if want:
        log(f"  weather forecasts for {len(out)} of {len(want)} outdoor games" + (f" (forecast service unreachable: {down}; average wind used)" if down else ""))
    return out

# ---------------- live odds from several books (The Odds API, free key) ----------------
URL_ODDS = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={key}&bookmakers={books}"
            "&markets=h2h,spreads,totals&oddsFormat=american&dateFormat=iso")
# Up to 20 books cost the same as two regions: 6 credits per build (free plan: 500 credits a month).
ODDS_BOOKS = ['pinnacle', 'draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'fanatics', 'espnbet', 'betrivers', 'hardrockbet',
              'ballybet', 'betparx', 'fliff', 'lowvig', 'betonlineag', 'bovada', 'mybookieag', 'betus']
ODDS_REF = 'pinnacle'      # fair prices: this book with its cut removed; the median of all books when it has no price
ODDS_REUSE_MIN = 15        # a build within this many minutes of the last odds download reuses it (saves credits)
ODDS_MAX_AGE_H = 24        # older saved odds are not shown unless --offline
NICK = {'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF', 'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN',
        'Browns': 'CLE', 'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GB', 'Texans': 'HOU', 'Colts': 'IND',
        'Jaguars': 'JAX', 'Chiefs': 'KC', 'Rams': 'LA', 'Chargers': 'LAC', 'Raiders': 'LV', 'Dolphins': 'MIA', 'Vikings': 'MIN',
        'Patriots': 'NE', 'Saints': 'NO', 'Giants': 'NYG', 'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', 'Seahawks': 'SEA',
        '49ers': 'SF', 'Buccaneers': 'TB', 'Titans': 'TEN', 'Commanders': 'WAS', 'Team': 'WAS'}

def team_abbr(name):
    return NICK.get((name or '').split(' ')[-1])

def odds_key_file():
    """The Odds API key: the first word of nfl-odds-key.txt next to this script (or set ODDS_API_KEY)."""
    p = os.path.join(BASE, 'nfl-odds-key.txt')
    if os.path.exists(p):
        with open(p) as fh:
            words = fh.read().split()
        return words[0] if words else ''
    return ''

def fetch_odds():
    """Pregame prices from several books. Returns (saved dict or None, note for the card).
    The saved copy (cache/odds.json) holds the prices and the download time, never the key."""
    path = os.path.join(CACHE, 'odds.json')
    saved = None
    if os.path.exists(path):
        try:
            with open(path) as fh: saved = json.load(fh)
        except (ValueError, OSError):
            saved = None
    age_h = None
    if saved and saved.get('fetched_utc'):
        t = datetime.datetime.fromisoformat(saved['fetched_utc'])
        age_h = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 3600
    if '--no-odds' in ARGS:
        return None, 'skipped (--no-odds)'
    key = os.environ.get('ODDS_API_KEY', '').strip() or odds_key_file()
    if '--offline' in ARGS or not key:
        if saved and ('--offline' in ARGS or age_h < ODDS_MAX_AGE_H):
            log(f"  odds: using saved prices from {age_h:.1f} hours ago")
            return saved, 'saved'
        if not key:
            log("  odds: no key (add nfl-odds-key.txt, or the ODDS_API_KEY secret on GitHub, for live prices; see README)")
            return None, 'no key'
        return None, 'offline'
    if saved and age_h is not None and age_h * 60 < ODDS_REUSE_MIN:
        log(f"  odds: reusing prices from {age_h * 60:.0f} minutes ago (saves credits)")
        return saved, 'saved'
    url = URL_ODDS.format(key=key, books=','.join(ODDS_BOOKS))
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'nfl-build/1.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            events = json.loads(r.read().decode('utf-8'))
            left, used = r.headers.get('x-requests-remaining'), r.headers.get('x-requests-used')
        if not isinstance(events, list):
            raise ValueError('unexpected reply')
        saved = dict(fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                     remaining=left, used=used, events=events)
        with open(path + '.part', 'w') as fh: json.dump(saved, fh, separators=(',', ':'))
        os.replace(path + '.part', path)
        log(f"  odds: {len(events)} games from The Odds API; {left if left is not None else '?'} credits left this month")
        return saved, 'live'
    except urllib.error.HTTPError as e:
        why = {401: 'key rejected: check the key (nfl-odds-key.txt, or the ODDS_API_KEY secret on GitHub)',
               429: 'out of credits or too many requests'}.get(e.code, f'HTTP {e.code}')
    except Exception as e:
        why = f'unreachable ({e})'
    if saved and age_h is not None and age_h < ODDS_MAX_AGE_H:
        log(f"  odds: {why}; using saved prices from {age_h:.1f} hours ago")
        return saved, 'saved'
    log(f"  odds: {why}; card built without live prices")
    return None, why

def parse_odds(events, season_games):
    """game_id -> {book: {'ml': [home, away], 'sp': [home point, home price, away point, away price],
    'to': [over point, over price, under point, under price]}}, for games that have not started. Plus book titles."""
    now = datetime.datetime.now(datetime.timezone.utc)
    by_pair = {}
    for g in season_games:
        if g['result'] is None:
            by_pair.setdefault((g['home'], g['away']), []).append(g)
    def price(o):
        v = o.get('price')
        return int(round(v)) if isinstance(v, (int, float)) and abs(v) >= 100 else None
    def point(o):
        v = o.get('point')
        return float(v) if isinstance(v, (int, float)) else None
    out, titles = {}, {}
    for ev in events or []:
        try:
            kick = datetime.datetime.fromisoformat(ev['commence_time'].replace('Z', '+00:00'))
        except (KeyError, ValueError, AttributeError):
            continue
        if kick <= now:
            continue                                  # started: in-game prices are not pregame prices
        h, a = team_abbr(ev.get('home_team')), team_abbr(ev.get('away_team'))
        hn, an = ev.get('home_team'), ev.get('away_team')
        cands = by_pair.get((h, a))
        if not cands and by_pair.get((a, h)):         # neutral-site game listed the other way round
            cands = by_pair[(a, h)]; hn, an = an, hn
        if not cands:
            continue
        kd = (kick - datetime.timedelta(hours=5)).date()
        g = min(cands, key=lambda x: abs((datetime.date.fromisoformat(x['gameday']) - kd).days))
        if abs((datetime.date.fromisoformat(g['gameday']) - kd).days) > 3:
            continue
        gb = {}
        for bk in ev.get('bookmakers') or []:
            key = bk.get('key')
            if not key:
                continue
            o = {}
            for mk in bk.get('markets') or []:
                oc = {x.get('name'): x for x in mk.get('outcomes') or []}
                if mk.get('key') == 'h2h' and hn in oc and an in oc:
                    v = [price(oc[hn]), price(oc[an])]
                    if None not in v: o['ml'] = v
                elif mk.get('key') == 'spreads' and hn in oc and an in oc:
                    v = [point(oc[hn]), price(oc[hn]), point(oc[an]), price(oc[an])]
                    if None not in v: o['sp'] = v
                elif mk.get('key') == 'totals' and 'Over' in oc and 'Under' in oc:
                    v = [point(oc['Over']), price(oc['Over']), point(oc['Under']), price(oc['Under'])]
                    if None not in v: o['to'] = v
            if o:
                gb[key] = o; titles[key] = bk.get('title') or key
        if gb:
            out[g['game_id']] = gb
    return out, titles

def median(xs):
    xs = sorted(xs); n = len(xs)
    return None if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)

def odds_reference(gb):
    """The market's number for one game from live prices: the reference book when it has the market, else the median.
    Returns home win probability (cut removed), home spread (positive = home favored) and total."""
    ref = {}
    ml = {k: no_vig(v['ml'][0], v['ml'][1]) for k, v in gb.items() if 'ml' in v}
    sp = {k: -v['sp'][0] for k, v in gb.items() if 'sp' in v}
    to = {k: v['to'][0] for k, v in gb.items() if 'to' in v and v['to'][0] == v['to'][2]}
    if ml:
        ref['p_home'] = ml[ODDS_REF] if ODDS_REF in ml else median(list(ml.values()))
    if sp:
        ref['spread'] = sp[ODDS_REF] if ODDS_REF in sp else round(median(list(sp.values())) * 2) / 2
    if to:
        ref['total'] = to[ODDS_REF] if ODDS_REF in to else round(median(list(to.values())) * 2) / 2
    ref['src'] = ODDS_REF if any(ODDS_REF in d for d in (ml, sp, to)) else 'median'
    return ref

ODDS_COLS = ['seen', 'game_id', 'book', 'market', 'v1', 'v2', 'v3', 'v4']

def track_odds(path, odds, stamp, final_ids):
    """Append each book's price whenever it changes (nfl-odds-seen.csv), so the card can show movement by book.
    Finished games keep only their first and last rows. Returns (game_id, book, market) -> (first values, first seen)."""
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as fh: rows = list(csv.DictReader(fh))
    last, first = {}, {}
    vals = lambda r: [fnum(r[c]) for c in ('v1', 'v2', 'v3', 'v4')]
    for r in rows:
        k = (r['game_id'], r['book'], r['market'])
        first.setdefault(k, (vals(r), r['seen'])); last[k] = vals(r)
    added = 0
    for gid, gb in odds.items():
        for book, o in gb.items():
            for mkt, v in o.items():
                v4 = [float(x) for x in v] + [None] * (4 - len(v))
                k = (gid, book, mkt)
                if last.get(k) != v4:
                    rows.append(dict(seen=stamp, game_id=gid, book=book, market=mkt,
                                     **{c: ('' if x is None else (int(x) if x == int(x) else x)) for c, x in zip(('v1', 'v2', 'v3', 'v4'), v4)}))
                    last[k] = v4; first.setdefault(k, (v4, stamp)); added += 1
    keep = []
    if final_ids:                                     # prune finished games to first and last row per book and market
        by = {}
        for i, r in enumerate(rows):
            if r['game_id'] in final_ids:
                by.setdefault((r['game_id'], r['book'], r['market']), []).append(i)
        drop = set(i for ix in by.values() for i in ix[1:-1])
        keep = [r for i, r in enumerate(rows) if i not in drop]
    else:
        keep = rows
    if added or len(keep) != len(rows):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=ODDS_COLS); w.writeheader()
            for r in keep: w.writerow({c: r.get(c, '') for c in ODDS_COLS})
    return first

def load_team_stats(current_season):
    """(game_id, team) -> raw stat row; history from data/team-stats.csv, this season from the downloaded weekly file."""
    ts = {}
    paths = [(os.path.join(DATA, 'team-stats.csv'), None)] + \
            [(os.path.join(CACHE, f'tw{y}.csv'), y) for y in range(SHIPPED_THROUGH + 1, current_season + 1)]
    for path, only in paths:
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                if not r['team'] or not r['game_id']: continue
                s = int(r['season'])
                if (only is None and s > SHIPPED_THROUGH) or (only is not None and s != only): continue
                ts[(r['game_id'], TEAM_FIX.get(r['team'], r['team']))] = r
    return ts

def team_obs(r, o):
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
    ob['cpoe_off'] = fnum(r.get('passing_cpoe'))
    ob['cpoe_def'] = fnum(o.get('passing_cpoe'))
    return ob

class Trackers:
    def __init__(self):
        self.ew, self.lg, self.season, self.last, self.streak, self.coach, self.qb_starts, self.hfa = {}, {}, {}, {}, {}, {}, {}, {}

    def get(self, t, k, h):
        v = self.ew.get((t, k, h))
        return self.lg.get(k, 0.0) if v is None else v

    def new_season(self, teams):
        for key in list(self.ew):
            m = self.lg.get(key[1], 0.0)
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
                a = 1 - 0.5 ** (1 / FHL[h])
                v = self.ew.get((t, k, h))
                self.ew[(t, k, h)] = x if v is None else v + a * (x - v)

def rich_features(games, base_feats, qbg, tstats, pred_by, keep_seasons, pbt=None, injmap=None, wx_by=None):
    """Feature dict for every game in keep_seasons, each a snapshot before that game's result."""
    base_by = {f['game']['game_id']: f for f in base_feats}
    season_teams = {}
    for g in games:
        season_teams.setdefault(g['season'], set()).update((g['home'], g['away']))
    T = Trackers(); cur = None; out = {}
    for g in games:
        gid, h, a = g['game_id'], g['home'], g['away']
        if g['season'] != cur:
            T.new_season(season_teams[g['season']]); cur = g['season']
        b = base_by[gid]
        if g['season'] in keep_seasons:
            f = {}
            for k, hs in EW.items():
                for hh in hs:
                    vh, va = T.get(h, k, hh), T.get(a, k, hh)
                    f[f'd_{k}_{hh}'] = vh - va
                    if f'{k}_{hh}' in SUM_KEYS:
                        f[f's_{k}_{hh}'] = vh + va
            for side, t in (('h', h), ('a', a)):
                sn = T.season.get(t, dict(w=0, l=0, gp=0, pd=0))
                f[f'{side}_gp'] = sn['gp']
                f[f'{side}_wpct'] = (sn['w'] + 1) / (sn['w'] + sn['l'] + 2)
                f[f'{side}_pd'] = sn['pd'] / (sn['gp'] + 2)
                L = T.last.get(t, {})
                f[f'{side}_last_margin'] = L.get('margin', 0.0); f[f'{side}_last_ats'] = L.get('ats', 0.0)
                f[f'{side}_last_epa'] = L.get('epa_pg', 0.0); f[f'{side}_last_to'] = L.get('to_margin', 0.0)
                ws, ats_s = T.streak.get(t, (0, 0))
                f[f'{side}_streak'] = ws; f[f'{side}_ats_streak'] = ats_s
                cn, cg = T.coach.get(t, ('', 0))
                coach_now = g['home_coach'] if side == 'h' else g['away_coach']
                tenure = cg if coach_now == cn else 0
                f[f'{side}_coach_games'] = min(tenure, 100); f[f'{side}_coach_new'] = 1.0 if tenure < 17 else 0.0
                st = T.qb_starts.get(b['qb_h' if side == 'h' else 'qb_a'], 0)
                f[f'{side}_qb_starts'] = min(st, 150); f[f'{side}_qb_rookie'] = 1.0 if st < 10 else 0.0
                f[f'{side}_hfa'] = T.hfa.get(t, 0.0)
            for k in ('gp', 'wpct', 'pd', 'last_margin', 'last_ats', 'last_epa', 'last_to', 'streak', 'ats_streak',
                      'coach_games', 'coach_new', 'qb_starts', 'qb_rookie'):
                f[f'd_{k}'] = f[f'h_{k}'] - f[f'a_{k}']
            f['elo_spread'] = b['elo_spread']; f['qa_h'] = b['qa_h']; f['qa_a'] = b['qa_a']; f['qa_diff'] = b['qa_h'] - b['qa_a']
            f['epa_net'] = b['epa_net']; f['oh'] = b['oh']; f['dh'] = b['dh']; f['oa'] = b['oa']; f['da'] = b['da']
            f['pace_sum'] = b['pace']; f['lg_total'] = b['lg_total']; f['bc_early'] = b['bc_early']; f['bc_late'] = b['bc_late']
            f['qb_new_h'] = 1.0 if b['qb_new_h'] else 0.0; f['qb_new_a'] = 1.0 if b['qb_new_a'] else 0.0
            f['elo_h'] = b['elo_h'] - 1500; f['elo_a'] = b['elo_a'] - 1500
            pw = pred_by.get(gid)
            f['pred'] = pw[0] if pw else b['elo_spread']; f['pred_total'] = pw[1] if pw else b['lg_total']
            f['has_pred'] = 1.0 if pw else 0.0
            f['week'] = g['week']; f['playoff'] = 0.0 if g['gtype'] == 'REG' else 1.0
            hour = int(g['gametime'][:2]) if g['gametime'][:2].isdigit() else 13
            f['prime'] = 1.0 if hour >= 19 else 0.0
            f['thu'] = 1.0 if g['weekday'] == 'Thursday' else 0.0; f['mon'] = 1.0 if g['weekday'] == 'Monday' else 0.0
            f['neutral'] = 1.0 if g['neutral'] else 0.0; f['div'] = float(g['div'])
            f['dome'] = 1.0 if g['roof'] in ('dome', 'closed') else 0.0
            f['open_roof'] = 1.0 if g['roof'] == 'open' else 0.0
            f['cold'] = 1.0 if (g['roof'] == 'outdoors' and h in COLD and int(g['gameday'][5:7]) in (11, 12, 1, 2)) else 0.0
            f['grass'] = 1.0 if g['surface'].strip() in ('grass', 'dessograss') else 0.0
            f['tzdiff'] = TZ.get(a, 0) - TZ.get(h, 0)
            f['rest_h'] = min(g['home_rest'], 14); f['rest_a'] = min(g['away_rest'], 14)
            f['rest_diff'] = f['rest_h'] - f['rest_a']
            f['short_h'] = 1.0 if g['home_rest'] <= 5 else 0.0; f['short_a'] = 1.0 if g['away_rest'] <= 5 else 0.0
            f['bye_h'] = 1.0 if (g['home_rest'] >= 13 and g['week'] > 1) else 0.0
            f['bye_a'] = 1.0 if (g['away_rest'] >= 13 and g['week'] > 1) else 0.0
            ih, ia = (injmap or {}).get((gid, h)), (injmap or {}).get((gid, a))
            f.update(injury_features(ih[0] if ih else None, ia[0] if ia else None))
            fw = (wx_by or {}).get(gid)
            f.update(weather_features(g['roof'], *(fw if fw else (g['temp'], g['wind']))))
            out[gid] = f
        if g['result'] is None:
            continue
        sp, tl, m = g['spread_line'], g['total_line'], g['result']
        rh, ra = tstats.get((gid, h)), tstats.get((gid, a))
        ph, pa = (pbt or {}).get((gid, h)), (pbt or {}).get((gid, a))
        for t, r, o, sgn, pp, pq in ((h, rh, ra, 1, ph, pa), (a, ra, rh, -1, pa, ph)):
            ob = (team_obs(r, o) if (r and o) else None) or {}
            if pp and pq:
                ob.update(pbp_obs(pp, pq))
            mm = sgn * m
            ob['margin'] = mm
            ob['pts_for'] = g['home_score'] if sgn == 1 else g['away_score']
            ob['pts_against'] = g['away_score'] if sgn == 1 else g['home_score']
            if 'epa_pg' in ob: ob['luck'] = mm - ob['epa_pg']
            if sp is not None: ob['ats'] = sgn * (m - sp)
            if tl is not None: ob['ou'] = g['total'] - tl
            ob['win'] = 1.0 if mm > 0 else (0.0 if mm < 0 else 0.5)
            T.update(t, ob)
            sn = T.season.setdefault(t, dict(w=0.0, l=0.0, gp=0, pd=0.0))
            sn['gp'] += 1; sn['pd'] += mm
            if mm > 0: sn['w'] += 1
            elif mm < 0: sn['l'] += 1
            else: sn['w'] += 0.5; sn['l'] += 0.5
            T.last[t] = dict(margin=mm, ats=ob.get('ats', 0.0), epa_pg=ob.get('epa_pg', 0.0), to_margin=ob.get('to_margin', 0.0))
            ws, a_s = T.streak.get(t, (0, 0))
            ws = (ws + 1 if ws > 0 else 1) if mm > 0 else ((ws - 1 if ws < 0 else -1) if mm < 0 else 0)
            if 'ats' in ob:
                a_s = (a_s + 1 if a_s > 0 else 1) if ob['ats'] > 0 else ((a_s - 1 if a_s < 0 else -1) if ob['ats'] < 0 else 0)
            T.streak[t] = (ws, a_s)
            coach_now = g['home_coach'] if sgn == 1 else g['away_coach']
            cn, cg = T.coach.get(t, ('', 0))
            T.coach[t] = (coach_now, cg + 1 if coach_now == cn else 1)
            q = qbg.get((gid, t))
            if q: T.qb_starts[q['name']] = T.qb_starts.get(q['name'], 0) + 1
        if not g['neutral']:
            T.hfa[h] = T.hfa.get(h, 0.0) + 0.05 * ((m - b['elo_spread']) - T.hfa.get(h, 0.0))
    return out

# ---------------- ratings pass ----------------
def run_ratings(games, tg, qbg, P):
    elo, off, dfn, pace, qbr, tqbr, last_qb = {}, {}, {}, {}, {}, {}, {}
    lg_epa, lg_total, lg_qb = 0.0, 44.0, 45.0
    a_epa = 1 - 0.5 ** (1 / P['EPA_HL']); a_pace = 1 - 0.5 ** (1 / P['PACE_HL'])
    cur_season = None
    feats = []
    for g in games:
        if g['season'] != cur_season:
            for t in elo: elo[t] = elo[t] * (1 - P['REGRESS']) + P['MEAN'] * P['REGRESS']
            for t in off: off[t] *= P['EPA_CARRY']; dfn[t] *= P['EPA_CARRY']
            for n in qbr: qbr[n] = qbr[n] * (1 - P['QB_OFF']) + lg_qb * P['QB_OFF']
            for t in tqbr: tqbr[t] = tqbr[t] * (1 - P['QB_OFF']) + lg_qb * P['QB_OFF']
            cur_season = g['season']
        h, a = g['home'], g['away']
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        hfa = 0.0 if g['neutral'] else P['HFA']
        rest_h = P['REST'] if (g['week'] > 1 and g['home_rest'] >= 13) else 0.0
        rest_a = P['REST'] if (g['week'] > 1 and g['away_rest'] >= 13) else 0.0
        # QB adjustment: listed starter vs what the team has been getting
        def qadj(team, name):
            nm = name or last_qb.get(team, '')
            prior = lg_qb - P['QB_PRIOR_GAP']
            qr = qbr.get(nm, prior); tr = tqbr.get(team, lg_qb)
            return P['QB_MULT'] * (qr - tr), nm, qr, nm not in qbr
        qa_h, name_h, qr_h, new_h = qadj(h, g['home_qb']); qa_a, name_a, qr_a, new_a = qadj(a, g['away_qb'])
        diff = (eh + hfa + rest_h + qa_h) - (ea + rest_a + qa_a)
        p_elo = 1 / (1 + 10 ** (-diff / 400))
        oh, dh, oa, da = off.get(h, 0.0), dfn.get(h, 0.0), off.get(a, 0.0), dfn.get(a, 0.0)
        hour = int(g['gametime'][:2]) if g['gametime'][:2].isdigit() else 13
        bc_early = 1 if (TZ.get(a, 0) <= -2 and hour <= 13 and not g['neutral']) else 0
        bc_late = 1 if (TZ.get(a, 0) == 0 and TZ.get(h, 0) <= -2 and hour >= 20 and not g['neutral']) else 0
        dome = 1 if g['roof'] in ('dome', 'closed') else 0
        f = dict(game=g, elo_spread=diff / 25, p_elo=p_elo, elo_h=eh, elo_a=ea, qa_h=qa_h / 25, qa_a=qa_a / 25,
                 qb_h=name_h, qb_a=name_a, qb_new_h=new_h, qb_new_a=new_a,
                 epa_net=(oh - dh) - (oa - da), oh=oh, dh=dh, oa=oa, da=da,
                 pace=pace.get(h, 63.0) + pace.get(a, 63.0), lg_total=lg_total,
                 rest_diff=max(-7, min(7, g['home_rest'] - g['away_rest'])) if g['week'] > 1 else 0,
                 div=g['div'], neutral=1 if g['neutral'] else 0, bc_early=bc_early, bc_late=bc_late, dome=dome,
                 usual_qb_h=last_qb.get(h, ''), usual_qb_a=last_qb.get(a, ''))
        feats.append(f)
        if g['result'] is None: continue
        # ---- updates ----
        m = g['result']; s = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
        th, ta = tg.get((g['game_id'], h)), tg.get((g['game_id'], a))
        if th and ta and th['plays'] > 20 and ta['plays'] > 20:
            epa_margin = (th['pepa'] + th['repa']) - (ta['pepa'] + ta['repa'])
            m_used = P['MOV_W'] * m + (1 - P['MOV_W']) * epa_margin
            raw_h, raw_a = (th['pepa'] + th['repa']) / th['plays'], (ta['pepa'] + ta['repa']) / ta['plays']
            adj_oh, adj_dh = (raw_h - lg_epa) - da, (raw_a - lg_epa) - oa
            adj_oa, adj_da = (raw_a - lg_epa) - dh, (raw_h - lg_epa) - oh
            off[h] = oh + a_epa * (adj_oh - oh); dfn[h] = dh + a_epa * (adj_dh - dh)
            off[a] = oa + a_epa * (adj_oa - oa); dfn[a] = da + a_epa * (adj_da - da)
            lg_epa += P['LG_A'] * ((raw_h + raw_a) / 2 - lg_epa)
            pace[h] = pace.get(h, 63.0) + a_pace * (th['plays'] - pace.get(h, 63.0))
            pace[a] = pace.get(a, 63.0) + a_pace * (ta['plays'] - pace.get(a, 63.0))
        else:
            m_used = m
        if m_used == 0 or (m > 0) != (m_used > 0): m_used = m
        wdiff = diff if m >= 0 else -diff
        mult = math.log(abs(m_used) + 1) * (2.2 / (wdiff * 0.001 + 2.2))
        shift = P['K'] * mult * (s - p_elo)
        elo[h] = eh + shift; elo[a] = ea - shift
        lg_total += P['LG_A'] * ((g['home_score'] + g['away_score']) - lg_total)
        for team, name in ((h, name_h), (a, name_a)):
            q = qbg.get((g['game_id'], team))
            if not q: continue
            v = qb_value(q); nm = q['name']
            qbr[nm] = qbr.get(nm, lg_qb - P['QB_PRIOR_GAP']) + P['QB_ALPHA'] * (v - qbr.get(nm, lg_qb - P['QB_PRIOR_GAP']))
            tqbr[team] = tqbr.get(team, lg_qb) + P['QB_ALPHA'] * (v - tqbr.get(team, lg_qb))
            lg_qb += 0.01 * (v - lg_qb)
            last_qb[team] = nm
    state = dict(elo=elo, off=off, dfn=dfn, qbr=qbr, tqbr=tqbr, lg_qb=lg_qb, lg_total=lg_total)
    return feats, state

# ---------------- least squares ----------------
def ols(X, y, ridge=1e-6):
    k = len(X[0]); A = [[0.0] * k for _ in range(k)]; b = [0.0] * k
    for row, yi in zip(X, y):
        for i in range(k):
            ri = row[i]
            if ri == 0.0: continue
            b[i] += ri * yi
            Ai = A[i]
            for j in range(k): Ai[j] += ri * row[j]
    for i in range(k): A[i][i] += ridge
    # gaussian elimination with partial pivoting
    M = [A[i] + [b[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12: continue
        for r in range(k):
            if r == c: continue
            fac = M[r][c] / M[c][c]
            if fac:
                Mr, Mc = M[r], M[c]
                for j in range(c, k + 1): Mr[j] -= fac * Mc[j]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]

def x_spread(f): return [1.0, f['elo_spread'], f['epa_net'], f['rest_diff'], f['div'], f['bc_early'], f['bc_late']]
def x_epa(f): return [1.0, f['epa_net'], f['neutral']]
def x_total(f): return [1.0, f['oh'], f['dh'], f['oa'], f['da'], f['pace'], f['lg_total'], f['dome'], f['div']]
def dot(b, x): return sum(bi * xi for bi, xi in zip(b, x))

def walk_forward(feats):
    """Fit on seasons < T, predict T, for T in WF_FROM..last complete. Returns per-game preds (list aligned with feats subset)."""
    done = [f for f in feats if f['game']['result'] is not None and f['game']['season'] >= FIT_FROM]
    last = max(f['game']['season'] for f in done)
    out = []
    for T in range(WF_FROM, last + 1):
        train = [f for f in done if f['game']['season'] < T]; test = [f for f in done if f['game']['season'] == T]
        if len(test) < 50: continue
        bs = ols([x_spread(f) for f in train], [f['game']['result'] for f in train])
        be = ols([x_epa(f) for f in train], [f['game']['result'] for f in train])
        bt = ols([x_total(f) for f in train], [f['game']['total'] for f in train])
        for f in test:
            out.append(dict(f=f, pred=dot(bs, x_spread(f)), pred_epa=dot(be, x_epa(f)), pred_total=dot(bt, x_total(f))))
    return out

def fit_scale(wf):
    best = None
    for s in [x / 4 for x in range(20, 60)]:
        ll = 0.0; n = 0
        for w in wf:
            m = w['f']['game']['result']
            if m == 0: continue
            p = 1 / (1 + math.exp(-w['pred'] / s)); p = min(max(p, 1e-6), 1 - 1e-6)
            ll -= math.log(p) if m > 0 else math.log(1 - p); n += 1
        if best is None or ll / n < best[0]: best = (ll / n, s)
    return best[1], best[0]

def fit_logit(xs, ys, iters=40):
    """2-parameter logistic fit by Newton's method: P(y=1) = 1/(1+exp(-(a+b*x)))."""
    a = b = 0.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(a + b * x))); w = p * (1 - p)
            g0 += y - p; g1 += (y - p) * x; h00 += w; h01 += w * x; h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if det <= 0: break
        da = (h11 * g0 - h01 * g1) / det; db = (-h01 * g0 + h00 * g1) / det
        a += da; b += db
        if abs(da) + abs(db) < 1e-9: break
    return a, b

def calibrate(wf):
    """Map (model - close) disagreement to realized cover / over rates. This is the edge you can act on."""
    xs, ys = [], []
    for w in wf:
        g = w['f']['game']
        if g['spread_line'] is None or g['result'] == g['spread_line']: continue
        xs.append(w['pred'] - g['spread_line']); ys.append(1.0 if g['result'] > g['spread_line'] else 0.0)
    cs = fit_logit(xs, ys)
    xt, yt = [], []
    for w in wf:
        g = w['f']['game']
        if g['total_line'] is None or g['total'] == g['total_line']: continue
        xt.append(w['pred_total'] - g['total_line']); yt.append(1.0 if g['total'] > g['total_line'] else 0.0)
    ct = fit_logit(xt, yt)
    return dict(spread=[round(cs[0], 5), round(cs[1], 5)], total=[round(ct[0], 5), round(ct[1], 5)], n_spread=len(xs), n_total=len(xt))

def summarize(wf, scale):
    """Backtest metrics by season + overall."""
    def block(rows):
        n = len(rows); mae_m = sum(abs(w['pred'] - w['f']['game']['result']) for w in rows) / n
        mae_t = sum(abs(w['pred_total'] - w['f']['game']['total']) for w in rows) / n
        lined = [w for w in rows if w['f']['game']['spread_line'] is not None]
        mae_c = sum(abs(w['f']['game']['spread_line'] - w['f']['game']['result']) for w in lined) / max(1, len(lined))
        tl = [w for w in rows if w['f']['game']['total_line'] is not None]
        mae_ct = sum(abs(w['f']['game']['total_line'] - w['f']['game']['total']) for w in tl) / max(1, len(tl))
        ats = {}
        for th in (1, 2, 3, 4):
            W = L = 0
            for w in lined:
                g = w['f']['game']; d = w['pred'] - g['spread_line']
                if d >= th: W += g['result'] > g['spread_line']; L += g['result'] < g['spread_line']
                elif d <= -th: W += g['result'] < g['spread_line']; L += g['result'] > g['spread_line']
            ats[th] = [W, L]
        ou = {}
        for th in (2, 3, 4):
            W = L = 0
            for w in tl:
                g = w['f']['game']; d = w['pred_total'] - g['total_line']
                if d >= th: W += g['total'] > g['total_line']; L += g['total'] < g['total_line']
                elif d <= -th: W += g['total'] < g['total_line']; L += g['total'] > g['total_line']
            ou[th] = [W, L]
        # log loss: model vs market-implied (moneyline)
        llm = llk = 0.0; nl = 0
        for w in rows:
            g = w['f']['game']
            if g['home_ml'] is None or g['away_ml'] is None or g['result'] == 0: continue
            def imp(ml): return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
            ph, pa = imp(g['home_ml']), imp(g['away_ml']); pk = ph / (ph + pa)
            pm = 1 / (1 + math.exp(-w['pred'] / scale))
            y = 1 if g['result'] > 0 else 0
            llm -= math.log(pm if y else 1 - pm); llk -= math.log(pk if y else 1 - pk); nl += 1
        return dict(n=n, mae_model=round(mae_m, 2), mae_close=round(mae_c, 2), n_lined=len(lined),
                    mae_total_model=round(mae_t, 2), mae_total_close=round(mae_ct, 2), ats=ats, ou=ou,
                    logloss_model=round(llm / nl, 4) if nl else None, logloss_market=round(llk / nl, 4) if nl else None)
    seasons = sorted(set(w['f']['game']['season'] for w in wf))
    return dict(overall=block(wf), by_season={s: block([w for w in wf if w['f']['game']['season'] == s]) for s in seasons})

# ---------------- probabilities from empirical residuals ----------------
def cover_prob(pred, line, resid):
    """P(home covers), P(push) using integer margins = round(pred + r)."""
    W = Pu = 0
    for r in resid:
        m = int(math.floor(pred + r + 0.5))
        if m > line: W += 1
        elif m == line: Pu += 1
    n = len(resid)
    return W / n, Pu / n

# ---------------- season simulation ----------------
def simulate(season_games, p_home_by, n_sims=2000, seed=7):
    rnd = random.Random(seed)
    teams = sorted(set([g['home'] for g in season_games] + [g['away'] for g in season_games]))
    reg = [g for g in season_games if g['gtype'] == 'REG']
    base = {t: 0.0 for t in teams}
    for g in reg:
        if g['result'] is None: continue
        if g['result'] > 0: base[g['home']] += 1
        elif g['result'] < 0: base[g['away']] += 1
        else: base[g['home']] += 0.5; base[g['away']] += 0.5
    todo = [(g['home'], g['away'], p_home_by[g['game_id']]) for g in reg if g['result'] is None]
    wins_acc = {t: [] for t in teams}; div_w = {t: 0 for t in teams}; po = {t: 0 for t in teams}
    for _ in range(n_sims):
        w = dict(base)
        for h, a, p in todo:
            if rnd.random() < p: w[h] += 1
            else: w[a] += 1
        for t in teams: wins_acc[t].append(w[t])
        for conf in ('AFC', 'NFC'):
            winners = []
            for dv in ('East', 'North', 'South', 'West'):
                members = [t for t in teams if DIV.get(t) == f'{conf} {dv}']
                if not members: continue
                best = max(members, key=lambda t: (w[t], rnd.random())); winners.append(best); div_w[best] += 1
            rest = sorted([t for t in teams if DIV.get(t, '').startswith(conf) and t not in winners], key=lambda t: (w[t], rnd.random()), reverse=True)[:3]
            for t in winners + rest: po[t] += 1
    out = []
    for t in teams:
        ws = sorted(wins_acc[t]); n = len(ws)
        out.append(dict(team=t, div=DIV.get(t, ''), record_now=base[t], exp_wins=round(sum(ws) / n, 1), p10=ws[n // 10], p90=ws[9 * n // 10],
                        p_div=round(div_w[t] / n_sims, 3), p_playoff=round(po[t] / n_sims, 3)))
    out.sort(key=lambda r: -r['exp_wins'])
    return out

# ---------------- picks log ----------------
PICK_COLS = ['logged', 'season', 'week', 'game_id', 'away', 'home', 'type', 'side', 'line_at_pick', 'model_prob', 'edge',
             'status', 'result', 'close_line', 'clv', 'reason']
STRONG_LEAN = 4.0          # spread disagreements this large are logged as paper picks to track their live record

def now_et():
    """Current US Eastern time, whatever the device's time zone (schedule kickoff times are Eastern)."""
    return et_from_utc(datetime.datetime.now(datetime.timezone.utc))

def et_from_utc(utc):
    utc = utc.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    mar = datetime.datetime(utc.year, 3, 8); nov = datetime.datetime(utc.year, 11, 1)
    start = mar + datetime.timedelta(days=(6 - mar.weekday()) % 7, hours=7)      # second Sunday of March, 2am EST
    end = nov + datetime.timedelta(days=(6 - nov.weekday()) % 7, hours=6)        # first Sunday of November, 2am EDT
    return utc - datetime.timedelta(hours=4 if start <= utc < end else 5)

def track_lines(path, season_games):
    """First line seen for each upcoming game, so the card can show how the market has moved since."""
    seen = {}
    if os.path.exists(path):
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                seen[r['game_id']] = r
    added = 0
    for g in season_games:
        if g['result'] is None and g['spread_line'] is not None and g['game_id'] not in seen:
            seen[g['game_id']] = dict(game_id=g['game_id'], first_seen=now_et().date().isoformat(), spread=g['spread_line'],
                                      total='' if g['total_line'] is None else g['total_line'],
                                      home_ml='' if g['home_ml'] is None else g['home_ml'], away_ml='' if g['away_ml'] is None else g['away_ml'])
            added += 1
    if added:
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['game_id', 'first_seen', 'spread', 'total', 'home_ml', 'away_ml']); w.writeheader()
            for r in seen.values():
                w.writerow(r)
    return seen
def update_picks(path, games_by_id, new_picks):
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as fh: rows = list(csv.DictReader(fh))
    have = set((r['game_id'], r['type']) for r in rows)
    for r in rows:
        if r['status'] != 'open': continue
        g = games_by_id.get(r['game_id'])
        if not g or g['result'] is None: continue
        L = float(r['line_at_pick'])
        if r['type'] == 'spread':
            close = g['spread_line']; m = g['result']
            if r['side'] == g['home']: res = 'W' if m > L else ('P' if m == L else 'L'); clv = (close - L) if close is not None else ''
            else: res = 'W' if m < L else ('P' if m == L else 'L'); clv = (L - close) if close is not None else ''
        else:
            close = g['total_line']; t = g['total']
            if r['side'] == 'over': res = 'W' if t > L else ('P' if t == L else 'L'); clv = (close - L) if close is not None else ''
            else: res = 'W' if t < L else ('P' if t == L else 'L'); clv = (L - close) if close is not None else ''
        r.update(status='graded', result=res, close_line='' if close is None else close, clv='' if clv == '' else round(clv, 1))
    for p in new_picks:
        if (p['game_id'], p['type']) in have: continue
        rows.append(dict(logged=now_et().date().isoformat(), status='open', result='', close_line='', clv='', **p))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=PICK_COLS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, '') for k in PICK_COLS})
    return rows

# ---------------- main ----------------
def american_prob(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)

def no_vig(h, a):
    """Home win probability from two moneylines with the book's cut removed by the power method: p = q^k with k set so
    the two sides sum to one. Unlike dividing by the total, it leaves longshots their historical win rates
    (1999-2025 underdogs priced under 15%: won 8.9%, power method 10.3%, proportional method 12.0%)."""
    qh, qa = american_prob(h), american_prob(a)
    lo, hi = 0.5, 2.0
    for _ in range(50):
        k = (lo + hi) / 2
        if qh ** k + qa ** k > 1: lo = k
        else: hi = k
    return qh ** ((lo + hi) / 2)

def market_prob(g, s2l):
    """Home win probability the market implies, with the book's cut removed: from the two moneylines when posted,
    otherwise from the spread (mapping fitted on 2006-15). None when there is no line yet."""
    if g['home_ml'] and g['away_ml']:
        return no_vig(g['home_ml'], g['away_ml'])
    if g['spread_line'] is not None:
        return 1 / (1 + math.exp(-s2l * g['spread_line']))
    return None

def fair_price(p):
    """American odds that break even at win probability p."""
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)

def load_model():
    path = os.path.join(DATA, 'model.json')
    if not os.path.exists(path):
        sys.exit("data/model.json is missing; it ships with the package (built by lab/export.py)")
    with open(path) as fh:
        return json.load(fh)

def apply_model(M, f):
    """Standardize one game's features and apply the shipped coefficients.
    Returns home margin, total, home-win log-odds, and per-group contributions for margin and total."""
    clip = M['clip']
    z = []
    for n, mu, sd in zip(M['features'], M['mu'], M['sd']):
        x = f.get(n)
        z.append(0.0 if x is None else max(-clip, min(clip, (x - mu) / sd)))
    out = {}
    for key in ('margin', 'total', 'win'):
        c = M[key]['coef']
        out[key] = M[key]['intercept'] + sum(ci * zi for ci, zi in zip(c, z))
    for key in ('margin', 'total'):
        g = {}
        for n, ci, zi in zip(M['features'], M[key]['coef'], z):
            grp = M['groups'][n]
            g[grp] = g.get(grp, 0.0) + ci * zi
        out['groups_' + key] = g
    return out

def main():
    t0 = time.time()
    log("Loading games...")
    games = load_games()
    current_season = max(g['season'] for g in games if g['result'] is not None)
    log(f"  {len(games)} games; current season {current_season}")
    tg, qbg = load_stats(current_season)
    log(f"  team-game rows {len(tg)}, QB-game rows {len(qbg)}")

    if '--tune' in ARGS:
        tune(games, tg, qbg); return

    log("Ratings pass...")
    feats, state = run_ratings(games, tg, qbg, P)
    # ratings-only blend, fit on completed seasons before this one (an input to the full model, and shown for reference)
    prev = [f for f in feats if f['game']['result'] is not None and FIT_FROM <= f['game']['season'] < current_season]
    bs = ols([x_spread(f) for f in prev], [f['game']['result'] for f in prev])
    be = ols([x_epa(f) for f in prev], [f['game']['result'] for f in prev])
    btot = ols([x_total(f) for f in prev], [f['game']['total'] for f in prev])
    season_feats = [f for f in feats if f['game']['season'] == current_season]
    pred_by = {f['game']['game_id']: (dot(bs, x_spread(f)), dot(btot, x_total(f))) for f in season_feats}

    log("Feature pass...")
    tstats = load_team_stats(current_season)
    pbt = load_pbp_team(current_season)
    inj_path = os.path.join(CACHE, f'inj{current_season}.csv')
    have_inj = fetch_optional(URL_INJ.format(s=current_season), inj_path)
    snap_paths = []
    for yy in (current_season - 1, current_season):
        p = os.path.join(CACHE, f'snap{yy}.csv')
        if fetch_optional(URL_SNAP.format(s=yy), p, once=yy != current_season):
            snap_paths.append(p)
    season_games = [g for g in games if g['season'] == current_season]
    injmap = injury_impact(season_games, load_snaps(snap_paths), load_reports([inj_path] if have_inj else [])) if current_season >= INJ_FROM else {}
    wx_by = forecast_weather([g for g in season_games if g['result'] is None])
    saved_odds, odds_note = fetch_odds()
    odds, book_titles = parse_odds(saved_odds.get('events'), season_games) if saved_odds else ({}, {})
    odds_stamp = et_from_utc(datetime.datetime.fromisoformat(saved_odds['fetched_utc'])).strftime('%Y-%m-%d %H:%M') if saved_odds else None
    final_ids = set(g['game_id'] for g in season_games if g['result'] is not None)
    odds_first = track_odds(os.path.join(BASE, 'nfl-odds-seen.csv'), odds, odds_stamp, final_ids) if odds else {}
    if saved_odds:
        log(f"  odds: prices for {len(odds)} upcoming games from {len(book_titles)} books")
    rich = rich_features(games, feats, qbg, tstats, pred_by, {current_season}, pbt, injmap, wx_by)
    M = load_model()
    cal = M['calib']
    log(f"  model fit on {M['fit_seasons'][0]}-{M['fit_seasons'][1]}, {len(M['features'])} features; built {M['built']}")

    lines_seen = track_lines(os.path.join(BASE, 'nfl-lines-seen.csv'), season_games)
    weeks_played = [f['game']['week'] for f in season_feats if f['game']['result'] is not None]
    todo_weeks = [f['game']['week'] for f in season_feats if f['game']['result'] is None]
    current_week = min(todo_weeks) if todo_weeks else max(weeks_played)

    out_games, new_picks, p_home_by = [], [], {}
    for f in season_feats:
        g = f['game']; gid = g['game_id']
        fx = rich[gid]
        A = apply_model(M, fx)
        pred, ptot = A['margin'], A['total']
        sig = lambda x: 1 / (1 + math.exp(-max(-30, min(30, x))))
        p_home = sig(A['win'])                               # the model's own read (no market inputs)
        gb = odds.get(gid)
        ref = odds_reference(gb) if gb else {}              # live prices, when the build has them, beat the schedule's line
        mk = ref.get('spread', g['spread_line'])
        tl = ref.get('total', g['total_line'])
        # the pick: the market's favorite and its probability
        pm = ref['p_home'] if 'p_home' in ref else market_prob(dict(g, spread_line=mk), M['spread_to_logit'])
        if pm is not None:
            p_pick_home, basis = pm, 'market'
        else:                                               # no line posted yet: the model's own read
            p_pick_home, basis = p_home, 'model'
        p_home_by[gid] = p_pick_home
        winner = g['home'] if p_pick_home >= 0.5 else g['away']
        model_winner = g['home'] if p_home >= 0.5 else g['away']
        upset = mk is not None and mk != 0 and ((mk > 0) != (p_pick_home >= 0.5))
        top = lambda d: [[k, round(v, 2)] for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))]
        row = dict(game_id=gid, week=g['week'], gtype=g['gtype'], kickoff=f"{g['gameday']} {g['gametime']}", away=g['away'], home=g['home'],
                   away_qb=f['qb_a'], home_qb=f['qb_h'], usual_qb_away=f['usual_qb_a'], usual_qb_home=f['usual_qb_h'],
                   qb_new_away=f['qb_new_a'], qb_new_home=f['qb_new_h'], qb_adj_home=round(f['qa_h'], 1), qb_adj_away=round(f['qa_a'], 1),
                   spread_model=round(pred, 1), spread_blend=round(pred_by[gid][0], 1), spread_elo=round(f['elo_spread'], 1),
                   spread_epa=round(dot(be, x_epa(f)), 1), spread_market=mk, total_model=round(ptot, 1), total_blend=round(pred_by[gid][1], 1),
                   total_market=tl, p_home=round(p_home, 3), p_pick_home=round(p_pick_home, 3), pick_basis=basis,
                   winner=winner, p_winner=round(max(p_pick_home, 1 - p_pick_home), 3), upset=upset,
                   fair_home=fair_price(p_pick_home), fair_away=fair_price(1 - p_pick_home),
                   ml_home=g['home_ml'], ml_away=g['away_ml'],
                   model_winner=model_winner, p_model_winner=round(max(p_home, 1 - p_home), 3),
                   factors=top(A['groups_margin']), factors_total=top(A['groups_total']),
                   neutral=f['neutral'], dome=f['dome'], rest_diff=f['rest_diff'], div=f['div'], bc_early=f['bc_early'], bc_late=f['bc_late'],
                   final=g['result'] is not None, home_score=g['home_score'], away_score=g['away_score'],
                   elo_home=round(f['elo_h']), elo_away=round(f['elo_a']))
        # context for the card: injuries, weather, line movement
        for side, t in (('home', g['home']), ('away', g['away'])):
            ii = injmap.get((gid, t))
            row['inj_' + side] = [[p[1], p[2], p[3], p[4]] for p in ii[1][:3]] if ii else None
            row['injury_' + side] = round(sum(ii[0].values()), 2) if ii else None
        fw = wx_by.get(gid)
        if g['roof'] not in ('outdoors', 'open'):
            row['wx'] = dict(indoor=True)
        elif fw:
            row['wx'] = dict(indoor=False, temp=round(fw[0]), wind=round(fw[1]), src='forecast')
        elif g['temp'] is not None and g['wind'] is not None:
            row['wx'] = dict(indoor=False, temp=round(g['temp']), wind=round(g['wind']), src='game')
        else:
            row['wx'] = dict(indoor=False)
        ls = lines_seen.get(gid)
        if ls:
            row['spread_first'] = fnum(ls['spread']); row['total_first'] = fnum(ls['total']); row['first_seen'] = ls['first_seen']
        if gb:                                              # prices by book, and where each book opened when it differs
            row['odds'] = gb; row['ref_src'] = ref.get('src')
            opened = {}
            for book, o in gb.items():
                for mkt, v in o.items():
                    fv = odds_first.get((gid, book, mkt))
                    if fv and fv[0][:len(v)] != [float(x) for x in v]:
                        opened.setdefault(book, {})[mkt] = fv[0][:len(v)] + [fv[1][:10]]
            if opened:
                row['odds_open'] = opened
            rs, rt = odds_first.get((gid, ODDS_REF, 'sp')), odds_first.get((gid, ODDS_REF, 'to'))
            if rs:
                row['spread_first'] = -rs[0][0]; row['first_seen'] = rs[1][:10]
            if rt:
                row['total_first'] = rt[0][0]
        if g['result'] is not None and (g['home_spread_odds'] or g['over_odds']):
            row['close'] = dict(sp=[g['home_spread_odds'], g['away_spread_odds']], to=[g['over_odds'], g['under_odds']])
        # paper and edge picks for this week's games that have not kicked off
        kick = datetime.datetime.fromisoformat(f"{g['gameday']}T{g['gametime'][:5]}")
        open_now = g['result'] is None and g['week'] == current_week and kick > now_et()
        if mk is not None and open_now:
            p_cover = 1 / (1 + math.exp(-(cal['spread'][0] + cal['spread'][1] * (pred - mk))))
            eh, ea = p_cover - 0.5238, (1 - p_cover) - 0.5238
            strong = abs(pred - mk) >= STRONG_LEAN
            if max(eh, ea) >= EDGE_MIN or strong:
                home_side = (eh >= ea) if max(eh, ea) >= EDGE_MIN else (pred > mk)
                new_picks.append(dict(season=current_season, week=g['week'], game_id=gid, away=g['away'], home=g['home'], type='spread',
                                      side=g['home'] if home_side else g['away'], line_at_pick=mk,
                                      model_prob=round(p_cover if home_side else 1 - p_cover, 3), edge=round(eh if home_side else ea, 3),
                                      reason='edge' if max(eh, ea) >= EDGE_MIN else 'strong lean'))
        if tl is not None and open_now:
            p_over = 1 / (1 + math.exp(-(cal['total'][0] + cal['total'][1] * (ptot - tl))))
            eo, eu = p_over - 0.5238, (1 - p_over) - 0.5238
            if max(eo, eu) >= EDGE_MIN:
                new_picks.append(dict(season=current_season, week=g['week'], game_id=gid, away=g['away'], home=g['home'], type='total',
                                      side='over' if eo >= eu else 'under', line_at_pick=tl,
                                      model_prob=round(p_over if eo >= eu else 1 - p_over, 3), edge=round(max(eo, eu), 3), reason='edge'))
        out_games.append(row)

    games_by_id = {g['game_id']: g for g in games}
    picks = update_picks(os.path.join(BASE, 'nfl-picks.csv'), games_by_id, new_picks)
    log(f"  picks log: {len(picks)} rows ({sum(1 for p in picks if p['status'] == 'open')} open)")

    log("Season simulation...")
    sim = simulate([g for g in games if g['season'] == current_season], p_home_by)

    teams = sorted(state['elo'])
    ratings = [dict(team=t, elo=round(state['elo'][t]), off=round(state['off'].get(t, 0), 3), dfn=round(state['dfn'].get(t, 0), 3),
                    qb=None) for t in teams if t in DIV]
    payload = dict(built=now_et().strftime('%Y-%m-%d %H:%M'), season=current_season, current_week=current_week,
                   params=P, breakeven=0.5238, edge_min=EDGE_MIN, calib=cal, kelly=dict(frac=KELLY_FRAC, cap=KELLY_CAP),
                   inj_status=dict(have=bool(injmap), weeks=sorted(set(k[2] for k in load_reports([inj_path]))) if have_inj else []),
                   strong_lean=STRONG_LEAN,
                   model=dict(built=M['built'], fit_seasons=M['fit_seasons'], n_features=len(M['features']), spec=M['spec'],
                              intercept_margin=round(M['margin']['intercept'], 3), intercept_total=round(M['total']['intercept'], 3)),
                   games=out_games, residuals=M['residuals'], backtest=M['backtest'], ga=M['ga'], ga_ref=M.get('ga_ref', {}),
                   picks=[{k: r.get(k, '') for k in PICK_COLS} for r in picks], sim=sim, ratings=ratings, keys=M.get('keys'),
                   odds_note=odds_note,
                   odds_meta=dict(fetched=odds_stamp, ref=ODDS_REF, remaining=saved_odds.get('remaining'), n_games=len(odds),
                                  books=[[k, book_titles[k]] for k in ODDS_BOOKS if k in book_titles] +
                                        [[k, t] for k, t in sorted(book_titles.items()) if k not in ODDS_BOOKS]) if odds else None)
    tpl_path = os.path.join(BASE, 'dashboard-template.html')
    with open(tpl_path, encoding='utf-8') as fh: tpl = fh.read()
    html = tpl.replace('/*__DATA__*/', 'const DATA = ' + json.dumps(payload, separators=(',', ':')) + ';')
    out_path = os.path.join(BASE, 'nfl-dashboard.html')
    with open(out_path, 'w', encoding='utf-8') as fh: fh.write(html)
    log(f"Wrote {out_path} ({len(html) // 1024} KB) in {time.time() - t0:.1f}s")

def tune(games, tg, qbg):
    """Coordinate descent on walk-forward MAE (2016+). Slow-ish; run in a real computer, not the phone."""
    grid = dict(K=[14, 17, 20, 23, 26], HFA=[30, 40, 50, 60], REST=[0, 15, 25, 40], QB_MULT=[0, 2, 3.3, 4.5, 6],
                MOV_W=[1.0, 0.85, 0.7, 0.55, 0.4], EPA_HL=[5, 8, 12, 16], EPA_CARRY=[0.3, 0.45, 0.6, 0.75],
                QB_PRIOR_GAP=[10, 20, 30], QB_ALPHA=[0.06, 0.1, 0.15], REGRESS=[0.25, 0.333, 0.45])
    def score(Pn):
        feats, _ = run_ratings(games, tg, qbg, Pn)
        wf = [w for w in walk_forward(feats) if w['f']['game']['season'] >= 2016]
        return sum(abs(w['pred'] - w['f']['game']['result']) for w in wf) / len(wf)
    cur = dict(P); best = score(cur); log(f"start MAE {best:.4f}")
    for _round in range(2):
        for key, vals in grid.items():
            for v in vals:
                if v == cur[key]: continue
                Pn = dict(cur); Pn[key] = v; sc = score(Pn)
                if sc < best - 1e-5: best, cur = sc, Pn; log(f"  {key}={v} -> MAE {best:.4f}")
    log("TUNED:", json.dumps(cur)); log(f"final walk-forward MAE {best:.4f}")

if __name__ == '__main__':
    main()
