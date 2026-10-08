"""
proto_analyser.py
-----------------
[역할] 프로토 승부식: 모델 확률 × 베트맨 배당 → 기대가치(EV) 필터 → 켈리 기준 베팅 금액 → 1~2폴더 포트폴리오.

입력 CSV (long 포맷, 한 행 = 한 선택지):
    date, sport, league, home, away, market, line, selection, odds
  market / selection:
    WDL       H / D / A      (축구 승무패)
    ML        H / A          (야구·농구·배구 승패)
    HANDICAP  H / D / A      (line = 홈팀 핸디캡, 예: -1.5 / +1.0)
    OU        O / U          (line = 기준 점수, 예: 2.5 / 8.5 / 215.5)

확률 출처:
    WDL/ML        → 분류기 확률 (p_H, p_D, p_A)
    HANDICAP / OU → 몬테카를로 1만 회 시뮬레이션

수식:
    EV    = p × odds − 1
    Kelly = (p × odds − 1) / (odds − 1)
    베팅액 = bankroll × min(Kelly × KELLY_FRACTION, MAX_STAKE_RATIO)  → MIN_STAKE 단위 절사
    하루 총액 > bankroll × MAX_DAILY_EXPOSURE 이면 비례 축소

원칙: 다폴더 억제. 조합은 PROTO_MAX_FOLDERS(기본 2)까지만, 경기당 1픽.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

import config
from model import ModelFactory, apply_aliases

log = logging.getLogger(__name__)

ODDS_KEY = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}
MATCH_KEYS = ["date", "sport", "league", "home", "away"]


# ══════════════════════════════════════════════════════════════════════
# 기본 수식
# ══════════════════════════════════════════════════════════════════════
def expected_value(p: float, odds: float) -> float:
    return p * odds - 1


def kelly_fraction(p: float, odds: float) -> float:
    return max(0.0, (p * odds - 1) / (odds - 1)) if odds > 1 else 0.0


def stake_amount(kelly: float, bankroll: float) -> int:
    ratio = min(kelly * config.KELLY_FRACTION, config.MAX_STAKE_RATIO)
    return int(bankroll * ratio // config.MIN_STAKE * config.MIN_STAKE)


def overround(odds: list[float]) -> float:
    """배당 마진. 프로토는 해외 북메이커보다 마진이 커서 EV 확보가 어렵다."""
    return float(sum(1 / o for o in odds) - 1)


# ══════════════════════════════════════════════════════════════════════
# 확률 계산
# ══════════════════════════════════════════════════════════════════════
def _fixtures_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    """선택지 행들 → 경기 단위 fixtures. WDL/ML 배당은 모델의 배당 피처로 전달."""
    fx = df[MATCH_KEYS].drop_duplicates().reset_index(drop=True)
    main = df[df["market"].isin(["WDL", "ML"])]
    if not main.empty:
        piv = (main.assign(col=main["selection"].map(ODDS_KEY))
                   .pivot_table(index=MATCH_KEYS, columns="col", values="odds", aggfunc="first")
                   .reset_index())
        fx = fx.merge(piv, on=MATCH_KEYS, how="left")
    return fx


def attach_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 모델로 각 선택지의 확률(prob)을 계산해 붙인다."""
    out = []
    for sport, part in df.groupby("sport"):
        model = ModelFactory.load(sport)
        pred = model.predict(_fixtures_from_odds(part))
        part = part.merge(pred.drop(columns=[c for c in ODDS_KEY.values() if c in pred.columns]),
                          on=MATCH_KEYS, how="left")
        sims: dict[tuple, object] = {}
        probs = []
        for r in part.to_dict("records"):
            mkt, sel = r["market"], r["selection"]
            if mkt in ("WDL", "ML"):
                p = r.get(f"p_{sel}", np.nan)
            else:
                key = tuple(r[k] for k in MATCH_KEYS)
                if key not in sims:
                    sims[key] = model.simulate(r)
                sim = sims[key]
                p = (sim.handicap(float(r["line"])) if mkt == "HANDICAP"
                     else sim.over_under(float(r["line"])))[sel]
            probs.append(p)
        part["prob"] = probs
        out.append(part)
    return pd.concat(out, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# 단일 / 조합 분석
# ══════════════════════════════════════════════════════════════════════
def analyse_singles(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    df = df.copy()
    df["match"] = df["home"] + " vs " + df["away"]
    df["match_id"] = df.groupby(MATCH_KEYS, sort=False).ngroup()
    df["fair_odds"] = 1 / df["prob"].clip(lower=1e-9)
    df["ev"] = df["prob"] * df["odds"] - 1
    df["kelly"] = [kelly_fraction(p, o) for p, o in zip(df["prob"], df["odds"])]
    df["stake"] = [stake_amount(k, bankroll) for k in df["kelly"]]
    df["is_value"] = (df["ev"] >= config.MIN_EV) & (df["prob"] >= config.MIN_PROB)
    df["label"] = df.apply(lambda r: f"{r['match']} [{r['market']}{'' if pd.isna(r['line']) else ' ' + str(r['line'])}:{r['selection']}]", axis=1)
    return df.sort_values("ev", ascending=False).reset_index(drop=True)


def best_per_match(singles: pd.DataFrame) -> pd.DataFrame:
    """같은 경기의 여러 가치 픽은 상관관계가 크므로 EV 최상위 1개만 남긴다."""
    v = singles[singles["is_value"]]
    return v.loc[v.groupby("match_id")["ev"].idxmax()].sort_values("ev", ascending=False)


def analyse_combos(singles: pd.DataFrame, bankroll: float, top_n: int = 20) -> pd.DataFrame:
    pool = best_per_match(singles).head(15)
    rows = []
    for size in range(max(2, config.PROTO_MIN_FOLDERS), config.PROTO_MAX_FOLDERS + 1):
        for combo in itertools.combinations(pool.to_dict("records"), size):
            p = float(np.prod([c["prob"] for c in combo]))
            odds = float(np.prod([c["odds"] for c in combo]))
            k = kelly_fraction(p, odds)
            rows.append({"folders": size, "legs": " + ".join(c["label"] for c in combo),
                         "prob": p, "odds": round(odds, 2), "ev": expected_value(p, odds),
                         "kelly": k, "stake": stake_amount(k, bankroll)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # EV 단독 정렬은 고배당 조합을 과대평가 → 켈리(자금 성장률) 우선
    return out.sort_values(["kelly", "ev"], ascending=False).head(top_n).reset_index(drop=True)


def build_portfolio(singles: pd.DataFrame, combos: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """최종 추천: 단폴(허용 시) + 2폴더 상위, 하루 노출 한도 내로 비례 축소."""
    picks = []
    if config.PROTO_MIN_FOLDERS <= 1:
        for r in best_per_match(singles).to_dict("records"):
            picks.append({"type": "단폴", "bet": r["label"], "prob": r["prob"], "odds": r["odds"],
                          "ev": r["ev"], "kelly": r["kelly"], "stake": r["stake"]})
    for r in combos.head(5).to_dict("records") if not combos.empty else []:
        picks.append({"type": f"{r['folders']}폴더", "bet": r["legs"], "prob": r["prob"], "odds": r["odds"],
                      "ev": r["ev"], "kelly": r["kelly"], "stake": r["stake"]})
    pf = pd.DataFrame(picks)
    if pf.empty:
        return pf
    pf = pf[pf["stake"] > 0].sort_values("kelly", ascending=False).reset_index(drop=True)
    cap = bankroll * config.MAX_DAILY_EXPOSURE
    total = pf["stake"].sum()
    if total > cap:
        pf["stake"] = (pf["stake"] * cap / total // config.MIN_STAKE * config.MIN_STAKE).astype(int)
        pf = pf[pf["stake"] > 0]
    return pf


# ══════════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════════
def run(odds_df: pd.DataFrame, bankroll: float = config.BANKROLL):
    odds_df = odds_df.copy()
    if "line" not in odds_df.columns:
        odds_df["line"] = np.nan
    odds_df["date"] = pd.to_datetime(odds_df["date"], errors="coerce")
    odds_df["league"] = odds_df.get("league", pd.Series(index=odds_df.index, dtype=object)).fillna("")
    odds_df = apply_aliases(odds_df)
    priced = attach_probabilities(odds_df)
    singles = analyse_singles(priced, bankroll)
    combos = analyse_combos(singles, bankroll)
    portfolio = build_portfolio(singles, combos, bankroll)

    singles.to_csv(config.OUTPUT_DIR / "proto_singles.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(config.OUTPUT_DIR / "proto_combos.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(config.OUTPUT_DIR / "proto_portfolio.csv", index=False, encoding="utf-8-sig")
    log.info("프로토 분석: 가치픽 %d / 조합 %d / 최종 %d건, 총 %s원",
             int(singles["is_value"].sum()), len(combos), len(portfolio),
             f"{int(portfolio['stake'].sum()) if not portfolio.empty else 0:,}")
    return singles, combos, portfolio
