# Lab: how the model was searched and tested

Laptop only: needs Python 3 with numpy and scikit-learn (`pip install numpy scikit-learn`). Run from this folder, after `python3 ../nfl-build.py` has filled `../cache/`.

0. Raw inputs go in `raw/`: play-by-play (`raw/pbp/pbp{season}.csv.gz`), injury reports (`raw/inj/injuries_{season}.csv`) and snap counts (`raw/snap/snap_counts_{season}.csv`) from the nflverse releases. `python3 pbp.py` condenses play-by-play into `pbp-team.csv`.
1. `python3 features.py` builds `feats.csv`: about 200 pre-kickoff inputs for every game since 1999, each a snapshot taken before that game's result. `nfl-build.py` contains an exact copy of this code for the weekly run.
2. `python3 baselines.py` scores the market, the previous model and plain regularized models on the validation seasons.
3. Genetic searches. Split: fit on 2002–15, choose on 2016–20, never touch 2021–25.
   - `python3 gp.py --task ats --seed 1 --base --out runs/ats.json` evolves formula trees (tasks: `su_ind`, `su_mkt`, `ats`, `ou`). Add `--null` to run the identical search on outcomes shuffled within each season, which shows how much "accuracy" a search finds in pure noise.
   - `python3 gpr.py --task margin --seed 1 --base --out runs/margin.json` does the same for point spread and total (`margin`, `total`).
   - `python3 gafs.py --task su_ind --seed 1 --out runs/gafs.json` evolves which inputs a logistic model uses.
   - `python3 summarize.py` compares every run: accuracy on the searched seasons against accuracy on the unseen ones.
3b. `python3 evalfam.py 2016 2020` checks each feature family (play-by-play, injuries, weather) by walk-forward validation.
4. `python3 final.py` is the one-time test: yearly walk-forward refits scored on 2021–25 and the current season. Run it once, after every choice is made.
5. `python3 export.py` writes `../data/model.json`, which the weekly build reads. It includes the key-number pricing from `keys.py` (chance of each exact margin and total around the market's number, fitted on 2015 onward); `python3 keys.py` alone checks that fit on held-out seasons and refreshes it in place.

The runs behind the current model: formula-tree searches evaluated 40,000–80,000 candidates each, input-selection searches about 6,700. None beat the plain regularized model on unseen seasons, and against the spread and total the real searches did no better than their coin-flip controls, so `final.py` uses the plain models.
