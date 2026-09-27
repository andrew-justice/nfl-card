# NFL model card

A pandas-free Python model plus a self-contained HTML dashboard for weekly NFL winners, spreads and totals. Runs in a-Shell on iPhone or anywhere with Python 3.

## Run it on GitHub (no a-Shell)

GitHub can build the card on a schedule and host it at `https://YOUR-NAME.github.io/nfl-card/`. Add that page to your iPhone's Home Screen and it opens like an app, full screen, with the last copy kept for when you have no signal.

1. Put this folder in a public GitHub repository named `nfl-card` (free hosting needs a public repository; your odds key stays secret).
2. Settings, Secrets and variables, Actions, New repository secret: name `ODDS_API_KEY`, value your key from the-odds-api.com.
3. Settings, Pages, Build and deployment, Source: GitHub Actions.
4. Actions, Build card, Run workflow. About two minutes later the card is live at `https://YOUR-NAME.github.io/nfl-card/`.
5. On the iPhone, open that address in Safari, then Share, Add to Home Screen.

The schedule (`.github/workflows/build.yml`) builds every morning and about 30 minutes before each kickoff window: Thursday night, the three Sunday windows and Monday night. That's about 52 builds and 310 odds credits a month; Run workflow gives fresh prices any time. Each build saves the pick log and price history back to the repository. Bets you log are saved on the phone: Copy as text makes a backup, and Import moves bets between copies of the card (for example from the Claude app).

Free-plan limits: the repository must be public; scheduled runs can start a few minutes late when GitHub is busy; schedules pause after 60 days without activity (switch them back on under Actions); Pages is for personal, non-commercial sites.

## Weekly, in a-Shell

1. In a-Shell: `cd ~/Documents/nfl-model` and run `python3 nfl-build.py`. It downloads the schedule and lines, this season's weekly stats, play-by-play, injury reports and snap counts (about 5 MB early in the season, 25 MB by December), fetches weather forecasts for outdoor games and, if you've added an odds key, current prices from up to 17 sportsbooks, rebuilds ratings and features from 1999 forward, applies the fitted models in `data/model.json`, and writes `nfl-dashboard.html`. Under a minute early in the season, a minute or two late. Add `--offline` to skip downloads, `--no-weather` to skip forecasts, `--no-odds` to skip prices, `--quiet` to silence progress.
2. Open it with `internalbrowser file://$PWD/nfl-dashboard.html`. Tapping the file in the Files app shows it without running its scripts, so the page looks empty there.
3. Tap a game, type your book's line and price. Edge, stake and line shopping recompute. Numbers you type stay on the device when the browser allows it.

iOS Shortcut (a-Shell "Execute Command", two lines):
`cd ~/Documents/nfl-model && python3 nfl-build.py --quiet`
`cd ~/Documents/nfl-model && internalbrowser file://$PWD/nfl-dashboard.html`

Build twice a week: Tuesday or Wednesday for the new lines, and again Friday night or Saturday, after the final injury reports, for injury-adjusted numbers and closer weather forecasts. The card shows how each line has moved since the first build that saw it. With an odds key, build once more an hour or two before you bet: prices move, and the value list is only as fresh as the last build.

## Live odds (optional, free)

1. Get a free key at the-odds-api.com (free plan, 500 credits a month; they email the key).
2. In a-Shell: `cd ~/Documents/nfl-model` then `echo YOUR-KEY > nfl-odds-key.txt`.
3. Build as usual. The log shows how many games and books came back and how many credits are left. A build uses 6 credits, so the free plan covers about 80 builds a month; a build within 15 minutes of the last one reuses its prices and spends nothing.

The key stays in that file on your phone. The card never contains it, and `nfl-odds-key.txt` is not part of the package. The books are listed in `ODDS_BOOKS` at the top of `nfl-build.py`; up to 20 cost the same 6 credits.

## Reading the card

- **Winner picks** open the card: every game's pick, strongest first. The pick is the betting market's favorite with its chance to win after removing the book's cut, and its **fair price**, the moneyline that breaks even at that chance. "Toss-up" marks picks under 55%. "Ratings like X" marks games where the model on its own would pick the other team. Weeks without posted lines show the model's own pick and fair price, labeled "No line yet".
- **Injuries** show on each game when a regular is listed: the player, position and final status. Tap a game to see his share of snaps. The model weighs each player by his chance of missing the game and his snap share over the last four games.
- **Weather** shows when an outdoor game is forecast for 12+ mph wind or temperatures at or below 40°F. It comes from the free Open-Meteo forecast at build time; if the forecast can't be reached, the model uses average wind and the build still runs.
- **Moneyline check** (tap a game): both teams' chances and fair prices next to the best price at your books (or the market price without an odds key), plus a box for the price you're offered. The card shows the expected return per dollar and, when it's positive, a quarter-Kelly stake.
- **Prices at your books** (with an odds key): pick your sportsbooks under Books, and the card lists every bet at those books priced better than fair, with its expected return per dollar. Fair is Pinnacle's price with its cut removed; Pinnacle is a low-margin book whose prices are the usual benchmark for what a bet is worth. Tap a game for the best price at your books on every side, and "All books" for every book's line and where it opened. A winner pick shows "beats fair" when one of your books has a better price than fair. Spreads and totals at a different number are compared through the chance of every exact margin and total, so a half point through 3 or 7 is priced at what it's worth.
- **Bets** tab: log a bet from the value list, from a game or from the tab itself (game, bet, line, price, stake, book). The card grades it from the final score and prices it at the closing line: "worth +2.1% at closing prices" means the bet would have been a 2.1% edge at the price the market settled on. Over a few dozen bets, beating the close is a far earlier and steadier sign of a real edge than the win-loss record. In the Claude app the bets are saved to your account and only you can see them, and they grade once that card has been rebuilt with the final scores; in a-Shell they are saved in that browser, so use Copy as text now and then as a backup.
- **Against the line** lists every spread and total where the model sits 2 or more points from the market (4 or more is a strong lean), with the calibrated cover rate and edge. Copy sends picks and leans to the clipboard as text.
- Each game shows the pick, a projected score, then Model and Market face to face for spread and total; the bar between them is the gap in points.
- Tap a game for **what moves the number** (how much each family of inputs pushes the spread and total away from an average matchup), the outcome distributions with the market line drawn on top, the pieces of the number, your book's line and price, line shopping, and parlay legs.
- **Model-implied** probabilities come from the distribution of the model's own out-of-sample errors (so 3 and 7 are real and pushes are real). **Calibrated** probabilities are what actually happened, out of sample, when the model sat that far from the closing line. Edge, stakes, the pick log and parlay expectation use the calibrated number. Stakes are quarter-Kelly, capped at 2% of bankroll.
- **Season** tab: 2,000 simulated seasons from the pick probabilities; power ratings.
- **Record** tab: this season graded automatically (winners, spreads, totals), results by period with the locked test highlighted, the genetic search with its coin-flip control, the pick log, and season-by-season numbers.

## What the testing says

Every number below comes from models fitted only on earlier seasons. 2016–20 was used to choose between versions; 2021–25 was held back and scored once, at the end.

| 2021–25, locked test | Winners | Spread miss | Total miss |
|---|---|---|---|
| Betting favorite (the pick) / closing line | 66.5% | 9.76 | 10.31 |
| Model alone, with injuries, weather, play-by-play | 64.6% | 10.09 | 10.49 |
| Model alone, before those | 64.4% | 10.12 | 10.53 |
| Original model | 63.7% | 10.15 | 10.55 |

- Injuries, weather and play-by-play helped a little. On the 2016–20 validation seasons they lifted the model alone to 66.4% on winners, level with the favorite, but on 2021–25 the gain was 0.2 points. Most of what they add, the market already knows. (2021–25 has now been used to score two versions; no choice was made from it.)
- Picking winners: the favorite clears 65%. A model combining the line with the ratings picked the same winner 94% of the time and its probabilities were less accurate than the market's own, so the card's pick is the market. The model alone improved on the previous version and finished just under 65%.
- Betting favorites evenly does not make money: flat moneyline bets on every favorite returned −3.5% in 2021–25 and −3.0% over 2010–25, and every variation tested (top favorites only, favorite parlays) was negative over 2010–25.
- Against the line: the closing line still misses by less than the model in every period, and betting the model's side landed near 50% (301–314 at 2+ points of disagreement in 2021–25). Treat leans as numbers worth a second look, not as an edge.
- The genetic search (formula trees and input selection, about 700,000 candidate models across runs) found 56–58% against the spread on the seasons it searched, and the same 57–58% when fed coin-flip outcomes. On unseen seasons both fell to 47–52%. For winners and point predictions, no evolved formula beat the plain regularized model on unseen seasons, so the card uses the plain model.

## Using it to bet

The card is a price checker. Regular prices at any one book are almost always worse than fair, because the book's cut is built in. What beats fair is a book that lags the market, a better number at one book than the rest, and boosts and promotions. So:

1. Open accounts at two or three books and select them under Books; more books means more better-than-fair prices.
2. Bet only what shows in Prices at your books (or a boost that the moneyline check turns green), at the book named.
3. Log every bet. After 50 or more, look at "value at closing prices": consistently positive means the approach works even through a losing stretch; around zero or negative means it doesn't, whatever the record says.
4. Keep stakes flat and small, avoid parlays, and don't bet because the model disagrees with the line.

## Files

- `nfl-build.py` ratings, features and dashboard builder (pure Python). Rating constants at the top (`P`); `python3 nfl-build.py --offline --tune` re-tunes them on a laptop.
- `data/model.json` the fitted models (coefficients for 194 features, fitted on 2002–25), error distributions, calibration, backtest and search summaries, and the key-number pricing for spreads and totals. Built by `lab/export.py` (key numbers by `lab/keys.py`).
- `data/qb-games.csv`, `data/team-games.csv`, `data/team-stats.csv`, `data/pbp-team.csv` condensed history 1999–2025, shipped so the weekly run stays small. Later seasons are downloaded into `cache/` as needed.
- `nfl-lines-seen.csv` the first line the build saw for each game, for the line-movement notes.
- `nfl-odds-key.txt` your odds key (you create it; keep it private). `nfl-odds-seen.csv` every change in each book's price the builds have seen, for the "was" column; finished games keep only their first and last price. `cache/odds.json` the last prices downloaded.
- `dashboard-template.html` the page; the builder injects data at `/*__DATA__*/`.
- `.github/workflows/build.yml` the GitHub schedule and build; `web/` the Home Screen app pieces (manifest, icons, offline worker) and `site.py`, which turns the built card into the published site.
- `nfl-picks.csv` the pick log: every strong spread lean (4+ points off the line) is logged before kickoff as a paper pick, along with any spread or total whose calibrated edge reaches 1%; later builds grade them for result and closing line value. Don't edit it while a build runs.
- `cache/` downloaded files; safe to delete.
- `lab/` the research code (laptop only; needs numpy and scikit-learn): `features.py` builds the feature library, `gp.py`/`gpr.py` run the genetic programming search, `gafs.py` the input-selection search, `final.py` the walk-forward test, `export.py` writes `data/model.json`. See `lab/README.md`.

## Method, briefly

About 190 inputs per game, each measured before kickoff: Elo with margin of victory and a FiveThirtyEight-style quarterback adjustment; efficiency (EPA per play for passing and rushing, first-down and explosive-play rates, completion percentage over expected); recent results at three time scales; turnovers and how much of a margin came from turnover and kicking luck; pass rush and protection; special teams and penalties; quarterback and coaching experience; schedule, rest, travel and venue; injuries weighted by snap share; game-time weather; and play-by-play measures (success rate, early-down EPA, EPA in competitive game states, third down, red zone, pass rate over expected, special-teams EPA). Spread and total are ridge regressions on those inputs; the winner model is a ridge-regularized logistic regression. The pick is the market favorite, because on held-out seasons the market's own probability was the most accurate winner forecast available.

Pricing spreads and totals at other numbers: the chance of each exact margin is a normal curve around the market's number (13 points wide) multiplied by a weight for that margin, fitted on 2015–25 games, the seasons with the longer extra point. Margins of 3 land about three times as often as the curve alone says, 7 about twice, and 6, 10 and 14 more often too; totals get the same treatment. On held-out 2023–25 games this predicted exact margins much better than the curve alone (games landing on 3: 121 predicted, 127 actual; curve alone 46). Moving off 3 is worth about 18 cents a half point, off 7 about 11.

## Updating

Save the new `nfl-model.zip` into a-Shell's folder (Files app, On My iPhone, a-Shell), then in a-Shell: `cd ~/Documents && python3 -m zipfile -e nfl-model.zip .` It replaces the program, model and cache files and leaves your pick log, line history and odds key alone, since none of those are in the zip.

## Troubleshooting

- Download fails in a-Shell: the script prints a `curl -L -o ...` line for each file; run those, then `python3 nfl-build.py --offline`.
- `data/model.json is missing`: copy the whole folder again; the model file ships with the package.
- Everything in this folder avoids underscores in filenames on purpose.
- Data comes from the nflverse project (schedules, results, closing lines, weekly stats). Lines in the dataset are a consensus, not any one book; add an odds key for prices by book, or type your book's number into the card.
- Odds messages: "no key" means `nfl-odds-key.txt` is missing or empty; "key rejected" means the key is wrong; "out of credits" resets at the start of your monthly cycle. Any of these just leaves the prices off the card.
