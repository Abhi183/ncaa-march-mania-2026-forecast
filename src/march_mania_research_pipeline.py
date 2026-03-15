"""March Machine Learning Mania 2026 forecasting pipeline.

Research-oriented pipeline that:
1) builds team-season features from regular-season data,
2) computes Elo trajectories,
3) trains named HistGradientBoosting models for men and women,
4) writes Kaggle submission,
5) exports rich EDA and post-model visuals.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    name: str
    learning_rate: float = 0.03
    max_iter: int = 400
    max_depth: int = 4
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 42


def clip_preds(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 0.02, 0.98)


def compute_elo(compact_df: pd.DataFrame, k: int = 20, base_rating: int = 1500, hfa: int = 70) -> pd.DataFrame:
    records = []
    compact_df = compact_df.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    for season, sdf in compact_df.groupby("Season"):
        ratings: Dict[int, float] = {}
        for _, row in sdf.iterrows():
            w, l, loc = int(row["WTeamID"]), int(row["LTeamID"]), row["WLoc"]
            rw = ratings.get(w, float(base_rating))
            rl = ratings.get(l, float(base_rating))

            adj = hfa if loc == "H" else (-hfa if loc == "A" else 0)
            exp_w = 1.0 / (1.0 + 10 ** (-(rw + adj - rl) / 400.0))

            margin = int(row["WScore"]) - int(row["LScore"])
            mult = np.log(abs(margin) + 1.0) * (2.2 / ((rw - rl) * 0.001 + 2.2))
            change = k * mult * (1 - exp_w)

            ratings[w] = rw + change
            ratings[l] = rl - change

        for team, rating in ratings.items():
            records.append((season, team, rating))

    return pd.DataFrame(records, columns=["Season", "TeamID", "Elo"])


def add_both_sides_detailed(df: pd.DataFrame) -> pd.DataFrame:
    win = pd.DataFrame(
        {
            "Season": df["Season"],
            "DayNum": df["DayNum"],
            "TeamID": df["WTeamID"],
            "OppTeamID": df["LTeamID"],
            "Win": 1,
            "Score": df["WScore"],
            "OppScore": df["LScore"],
            "Loc": df["WLoc"],
            "NumOT": df["NumOT"],
            "FGM": df["WFGM"],
            "FGA": df["WFGA"],
            "FGM3": df["WFGM3"],
            "FGA3": df["WFGA3"],
            "FTM": df["WFTM"],
            "FTA": df["WFTA"],
            "OR": df["WOR"],
            "DR": df["WDR"],
            "Ast": df["WAst"],
            "TO": df["WTO"],
            "Stl": df["WStl"],
            "Blk": df["WBlk"],
            "PF": df["WPF"],
            "OppFGM": df["LFGM"],
            "OppFGA": df["LFGA"],
            "OppFGM3": df["LFGM3"],
            "OppFGA3": df["LFGA3"],
            "OppFTM": df["LFTM"],
            "OppFTA": df["LFTA"],
            "OppOR": df["LOR"],
            "OppDR": df["LDR"],
            "OppAst": df["LAst"],
            "OppTO": df["LTO"],
            "OppStl": df["LStl"],
            "OppBlk": df["LBlk"],
            "OppPF": df["LPF"],
        }
    )

    loc_swap = {"H": "A", "A": "H", "N": "N"}
    lose = pd.DataFrame(
        {
            "Season": df["Season"],
            "DayNum": df["DayNum"],
            "TeamID": df["LTeamID"],
            "OppTeamID": df["WTeamID"],
            "Win": 0,
            "Score": df["LScore"],
            "OppScore": df["WScore"],
            "Loc": df["WLoc"].map(loc_swap),
            "NumOT": df["NumOT"],
            "FGM": df["LFGM"],
            "FGA": df["LFGA"],
            "FGM3": df["LFGM3"],
            "FGA3": df["LFGA3"],
            "FTM": df["LFTM"],
            "FTA": df["LFTA"],
            "OR": df["LOR"],
            "DR": df["LDR"],
            "Ast": df["LAst"],
            "TO": df["LTO"],
            "Stl": df["LStl"],
            "Blk": df["LBlk"],
            "PF": df["LPF"],
            "OppFGM": df["WFGM"],
            "OppFGA": df["WFGA"],
            "OppFGM3": df["WFGM3"],
            "OppFGA3": df["WFGA3"],
            "OppFTM": df["WFTM"],
            "OppFTA": df["WFTA"],
            "OppOR": df["WOR"],
            "OppDR": df["WDR"],
            "OppAst": df["WAst"],
            "OppTO": df["WTO"],
            "OppStl": df["WStl"],
            "OppBlk": df["WBlk"],
            "OppPF": df["WPF"],
        }
    )

    games = pd.concat([win, lose], ignore_index=True)
    games["ScoreDiff"] = games["Score"] - games["OppScore"]
    games["IsHome"] = (games["Loc"] == "H").astype(int)
    games["IsAway"] = (games["Loc"] == "A").astype(int)
    games["IsNeutral"] = (games["Loc"] == "N").astype(int)

    games["Poss"] = games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"]
    games["OppPoss"] = games["OppFGA"] - games["OppOR"] + games["OppTO"] + 0.475 * games["OppFTA"]
    games["AvgPoss"] = ((games["Poss"] + games["OppPoss"]) / 2.0).clip(lower=1)

    games["eFG"] = (games["FGM"] + 0.5 * games["FGM3"]) / games["FGA"].replace(0, np.nan)
    games["Opp_eFG"] = (games["OppFGM"] + 0.5 * games["OppFGM3"]) / games["OppFGA"].replace(0, np.nan)
    games["TS"] = games["Score"] / (2 * (games["FGA"] + 0.44 * games["FTA"]).replace(0, np.nan))
    games["ORPct"] = games["OR"] / (games["OR"] + games["OppDR"]).replace(0, np.nan)
    games["DRPct"] = games["DR"] / (games["DR"] + games["OppOR"]).replace(0, np.nan)
    games["TOPct"] = games["TO"] / games["AvgPoss"]
    games["FTRate"] = games["FTA"] / games["FGA"].replace(0, np.nan)

    games["OffRtg"] = 100 * games["Score"] / games["AvgPoss"]
    games["DefRtg"] = 100 * games["OppScore"] / games["AvgPoss"]
    games["NetRtg"] = games["OffRtg"] - games["DefRtg"]
    return games


def build_team_features(prefix: str, reg_d: pd.DataFrame, reg_c: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    games = add_both_sides_detailed(reg_d).sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)
    feat = (
        games.groupby(["Season", "TeamID"])
        .agg(
            Games=("Win", "count"),
            Wins=("Win", "sum"),
            WinPct=("Win", "mean"),
            AvgScore=("Score", "mean"),
            AvgOppScore=("OppScore", "mean"),
            AvgMargin=("ScoreDiff", "mean"),
            MedianMargin=("ScoreDiff", "median"),
            AvgPoss=("AvgPoss", "mean"),
            OffRtg=("OffRtg", "mean"),
            DefRtg=("DefRtg", "mean"),
            NetRtg=("NetRtg", "mean"),
            eFG=("eFG", "mean"),
            Opp_eFG=("Opp_eFG", "mean"),
            TS=("TS", "mean"),
            ORPct=("ORPct", "mean"),
            DRPct=("DRPct", "mean"),
            TOPct=("TOPct", "mean"),
            FTRate=("FTRate", "mean"),
            Ast=("Ast", "mean"),
            TO=("TO", "mean"),
            Stl=("Stl", "mean"),
            Blk=("Blk", "mean"),
            PF=("PF", "mean"),
            HomeRate=("IsHome", "mean"),
            AwayRate=("IsAway", "mean"),
            NeutralRate=("IsNeutral", "mean"),
            OTs=("NumOT", "mean"),
        )
        .reset_index()
    )
    feat["Losses"] = feat["Games"] - feat["Wins"]

    recent = (
        games.groupby(["Season", "TeamID"], group_keys=False)
        .tail(10)
        .groupby(["Season", "TeamID"])
        .agg(
            Last10WinPct=("Win", "mean"),
            Last10Margin=("ScoreDiff", "mean"),
            Last10OffRtg=("OffRtg", "mean"),
            Last10DefRtg=("DefRtg", "mean"),
            Last10NetRtg=("NetRtg", "mean"),
        )
        .reset_index()
    )
    feat = feat.merge(recent, on=["Season", "TeamID"], how="left")

    opp_wpct = feat[["Season", "TeamID", "WinPct"]].rename(columns={"TeamID": "OppTeamID", "WinPct": "OppSeasonWinPct"})
    sched = games[["Season", "TeamID", "OppTeamID"]].merge(opp_wpct, on=["Season", "OppTeamID"], how="left")
    sched = sched.groupby(["Season", "TeamID"])["OppSeasonWinPct"].mean().reset_index().rename(columns={"OppSeasonWinPct": "SOS_WinPct"})
    feat = feat.merge(sched, on=["Season", "TeamID"], how="left")

    feat = feat.merge(compute_elo(reg_c), on=["Season", "TeamID"], how="left")

    if prefix == "M":
        massey = pd.read_csv(data_dir / "MMasseyOrdinals.csv")
        massey = massey[massey["RankingDayNum"] == 133].copy()
        m_avg = massey.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index().rename(columns={"OrdinalRank": "MasseyMeanRank"})
        m_med = massey.groupby(["Season", "TeamID"])["OrdinalRank"].median().reset_index().rename(columns={"OrdinalRank": "MasseyMedianRank"})
        feat = feat.merge(m_avg, on=["Season", "TeamID"], how="left").merge(m_med, on=["Season", "TeamID"], how="left")

    return feat


def build_matchups(tourney_df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    low = np.minimum(tourney_df["WTeamID"].values, tourney_df["LTeamID"].values)
    high = np.maximum(tourney_df["WTeamID"].values, tourney_df["LTeamID"].values)
    y = (tourney_df["WTeamID"].values == low).astype(int)

    base = pd.DataFrame({"Season": tourney_df["Season"].values, "Team1": low, "Team2": high, "Target": y})
    f1 = feat.rename(columns={c: f"T1_{c}" for c in feat.columns if c not in ["Season", "TeamID"]})
    f2 = feat.rename(columns={c: f"T2_{c}" for c in feat.columns if c not in ["Season", "TeamID"]})

    base = base.merge(f1, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    base = base.merge(f2, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])

    num_cols = [c for c in feat.columns if c not in ["Season", "TeamID"]]
    for col in num_cols:
        base[f"D_{col}"] = base[f"T1_{col}"] - base[f"T2_{col}"]
    return base


def build_submission_rows(sample_df: pd.DataFrame, feat: pd.DataFrame, prefix: str) -> pd.DataFrame:
    sub = sample_df.copy()
    parts = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["Team1"] = parts[1].astype(int)
    sub["Team2"] = parts[2].astype(int)

    sub = sub[sub["Team1"] < 3000].copy() if prefix == "M" else sub[sub["Team1"] >= 3000].copy()

    f1 = feat.rename(columns={c: f"T1_{c}" for c in feat.columns if c not in ["Season", "TeamID"]})
    f2 = feat.rename(columns={c: f"T2_{c}" for c in feat.columns if c not in ["Season", "TeamID"]})

    sub = sub.merge(f1, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    sub = sub.merge(f2, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])

    num_cols = [c for c in feat.columns if c not in ["Season", "TeamID"]]
    for col in num_cols:
        sub[f"D_{col}"] = sub[f"T1_{col}"] - sub[f"T2_{col}"]
    return sub


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("T1_") or c.startswith("T2_") or c.startswith("D_")]


def build_model(config: ModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=config.learning_rate,
                    max_iter=config.max_iter,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    l2_regularization=config.l2_regularization,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def train_and_predict(prefix: str, feat: pd.DataFrame, tourney_df: pd.DataFrame, sample_sub: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    train = build_matchups(tourney_df, feat)
    feature_cols = get_feature_columns(train)
    X, y = train[feature_cols], train["Target"].astype(int)
    model = build_model(config)

    valid_mask = train["Season"] >= (2021 if prefix == "M" else 2022)
    if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
        model.fit(X[~valid_mask], y[~valid_mask])
        val_pred = model.predict_proba(X[valid_mask])[:, 1]
        brier = brier_score_loss(y[valid_mask], val_pred)
        print(f"[{config.name}] {prefix} validation Brier score: {brier:.5f}")

    model.fit(X, y)
    sub_rows = build_submission_rows(sample_sub, feat, prefix)
    preds = model.predict_proba(sub_rows[feature_cols])[:, 1]
    out = pd.DataFrame({"ID": sub_rows["ID"], "Pred": clip_preds(preds)})
    return out, model, train


def save_eda_charts(mreg_c: pd.DataFrame, wreg_c: pd.DataFrame, mfeat: pd.DataFrame, wfeat: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    m_margin = mreg_c["WScore"] - mreg_c["LScore"]
    w_margin = wreg_c["WScore"] - wreg_c["LScore"]
    plt.hist(m_margin, bins=40, alpha=0.6, label="Men")
    plt.hist(w_margin, bins=40, alpha=0.6, label="Women")
    plt.title("Distribution of Victory Margins")
    plt.xlabel("Points")
    plt.ylabel("Game Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "margin_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    for df, label in [(mfeat, "Men"), (wfeat, "Women")]:
        season = df["Season"].max()
        vals = df[df["Season"] == season]["Elo"].dropna()
        plt.hist(vals, bins=30, alpha=0.6, label=f"{label} {season}")
    plt.title("Elo Distribution in Most Recent Season")
    plt.xlabel("Elo")
    plt.ylabel("Teams")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "elo_distribution_latest_season.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    latest_m = mfeat[mfeat["Season"] == mfeat["Season"].max()]
    latest_w = wfeat[wfeat["Season"] == wfeat["Season"].max()]
    plt.scatter(latest_m["OffRtg"], latest_m["DefRtg"], alpha=0.5, label="Men")
    plt.scatter(latest_w["OffRtg"], latest_w["DefRtg"], alpha=0.5, label="Women")
    plt.title("Offense-Defense Tradeoff")
    plt.xlabel("Offensive Rating")
    plt.ylabel("Defensive Rating (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "off_def_scatter.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run March Mania 2026 research pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/competitions/march-machine-learning-mania-2026"))
    parser.add_argument("--out-path", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--fig-dir", type=Path, default=Path("artifacts/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    mteams = pd.read_csv(data_dir / "MTeams.csv")
    wteams = pd.read_csv(data_dir / "WTeams.csv")
    mreg_c = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
    wreg_c = pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv")
    mreg_d = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    wreg_d = pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv")
    mtour = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")
    wtour = pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv")
    sample_sub = pd.read_csv(data_dir / "SampleSubmissionStage2.csv")

    print("Building team features...")
    mfeat = build_team_features("M", mreg_d, mreg_c, data_dir)
    wfeat = build_team_features("W", wreg_d, wreg_c, data_dir)

    men_cfg = ModelConfig(name="EloMomentum-GBDT-Men")
    women_cfg = ModelConfig(name="EloMomentum-GBDT-Women")

    print("Training models...")
    men_sub, _, _ = train_and_predict("M", mfeat, mtour, sample_sub, men_cfg)
    women_sub, _, _ = train_and_predict("W", wfeat, wtour, sample_sub, women_cfg)

    submission = pd.concat([men_sub, women_sub], ignore_index=True).sort_values("ID").reset_index(drop=True)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out_path, index=False)

    print(f"Saved submission to {args.out_path} ({len(submission)} rows)")
    save_eda_charts(mreg_c, wreg_c, mfeat, wfeat, args.fig_dir)
    print(f"Saved EDA charts to {args.fig_dir}")


if __name__ == "__main__":
    main()
