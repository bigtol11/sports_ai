"""
model.py
--------
[역할] 종목별 피처 생성 → XGBoost/LightGBM 앙상블 학습 → 승/무/패 확률 + 예상 득점 → 몬테카를로 시뮬레이션.

구성:
  1. FeatureBuilder        : 시간순 누적 상태(Elo·폼·롤링 스탯)로 '경기 전' 피처 생성 (데이터 누수 방지)
       └ 종목별 훅: SoccerFeatureBuilder(무승부 변수) / BaseballFeatureBuilder(선발투수 매치업)
                   BasketballFeatureBuilder(PACE·레이팅·쿼터) / VolleyballFeatureBuilder(세트 효율)
  2. EnsembleClassifier    : 결과 확률 (축구 H/D/A, 그 외 H/A)
     EnsembleRegressor     : 홈/원정 예상 득점 (몬테카를로 입력)
  3. MonteCarloSimulator   : 종목별 점수 분포로 1만 번 가상 경기 → 핸디캡/언오버/승1패/승5패/스페셜 확률
  4. SportModel            : 위 구성요소를 묶은 종목 단위 파이프라인 (train / predict / simulate / save / load)
     ModelFactory          : 종목명 → SportModel

배당이 있는 경기는 '배당 포함' 분류기, 없는 경기(토토 등)는 'Elo/폼 전용' 분류기를 자동 선택한다.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

import config

log = logging.getLogger(__name__)

ODDS_INPUT = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}


# ══════════════════════════════════════════════════════════════════════
# 1. 피처 엔지니어링
# ══════════════════════════════════════════════════════════════════════
def _form_deque() -> deque:
    return deque(maxlen=config.FORM_WINDOW)


def _rolling_dict() -> defaultdict:
    return defaultdict(_form_deque)   # 람다 대신 모듈 함수 → joblib 저장 가능


@dataclass
class TeamState:
    elo: float = config.ELO_INIT
    last_date: pd.Timestamp | None = None
    form: deque = field(default_factory=_form_deque)        # (pts, 득점, 실점)
    home_form: deque = field(default_factory=_form_deque)
    away_form: deque = field(default_factory=_form_deque)
    rolling: dict = field(default_factory=_rolling_dict)   # stat → 최근 N경기 값


def implied_probs(odds: list[float]) -> list[float]:
    """배당 → 북메이커 마진 제거 내재 확률."""
    arr = np.asarray(odds, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 1):
        return [np.nan] * len(arr)
    inv = 1 / arr
    return list(inv / inv.sum())


def _mean(values) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


class FeatureBuilder:
    """종목 공통 피처 빌더. 종목별 특성은 하위 클래스의 derive_match / extra_features 로 확장."""

    def __init__(self, sport: str):
        self.sport = sport
        self.spec = config.SPORTS[sport]
        self.outcomes = self.spec["outcomes"]
        self.teams: dict[str, TeamState] = defaultdict(TeamState)

    # ── 종목별 확장 훅 ────────────────────────────────────────────────
    def derive_match(self, row: dict) -> dict:
        """경기 후 파생 스탯 생성 (예: 축구 draw/total). row 를 수정해 반환."""
        return row

    def extra_features(self, feats: dict) -> dict:
        """기본 피처로부터 종목 특화 교차 피처 생성."""
        return feats

    # ── 피처 이름 ─────────────────────────────────────────────────────
    def feature_columns(self, with_odds: bool) -> list[str]:
        cols = ["elo_home", "elo_away", "elo_diff", "elo_exp_home", "elo_abs_diff",
                "rest_home", "rest_away", "b2b_home", "b2b_away"]
        if with_odds:
            cols += [f"imp_{o}" for o in self.outcomes]
        for side in ("h_all", "a_all", "h_venue", "a_venue"):
            cols += [f"{side}_pts", f"{side}_gf", f"{side}_ga"]
        cols += ["form_pts_diff", "gd_diff"]
        for s in self.spec["rolling_stats"]:
            cols += [f"h_{s}", f"a_{s}", f"diff_{s}"]
        for p in self.spec["prematch"]:
            cols += [f"home_{p}", f"away_{p}", f"diff_{p}"]
        cols += self.extra_feature_names()
        return cols

    def extra_feature_names(self) -> list[str]:
        return []

    # ── 경기 전 피처 계산 (상태 변경 없음) ────────────────────────────
    def features_for(self, row: dict) -> dict:
        h, a = self.teams[row["home"]], self.teams[row["away"]]
        date = row["date"]
        elo_diff = h.elo + self.spec["elo_home_adv"] - a.elo
        rest_h = (date - h.last_date).days if h.last_date is not None else np.nan
        rest_a = (date - a.last_date).days if a.last_date is not None else np.nan

        f = {
            "elo_home": h.elo, "elo_away": a.elo, "elo_diff": elo_diff,
            "elo_exp_home": 1 / (1 + 10 ** (-elo_diff / 400)), "elo_abs_diff": abs(elo_diff),
            "rest_home": rest_h, "rest_away": rest_a,
            "b2b_home": float(rest_h <= 1) if np.isfinite(rest_h) else np.nan,
            "b2b_away": float(rest_a <= 1) if np.isfinite(rest_a) else np.nan,
        }
        imps = implied_probs([row.get(ODDS_INPUT[o], np.nan) for o in self.outcomes])
        f.update({f"imp_{o}": p for o, p in zip(self.outcomes, imps)})

        for name, form in (("h_all", h.form), ("a_all", a.form),
                           ("h_venue", h.home_form), ("a_venue", a.away_form)):
            arr = np.array(form, dtype=float) if form else None
            f[f"{name}_pts"] = arr[:, 0].mean() if arr is not None else np.nan
            f[f"{name}_gf"] = arr[:, 1].mean() if arr is not None else np.nan
            f[f"{name}_ga"] = arr[:, 2].mean() if arr is not None else np.nan
        f["form_pts_diff"] = f["h_all_pts"] - f["a_all_pts"]
        f["gd_diff"] = (f["h_all_gf"] - f["h_all_ga"]) - (f["a_all_gf"] - f["a_all_ga"])

        for s in self.spec["rolling_stats"]:
            f[f"h_{s}"], f[f"a_{s}"] = _mean(h.rolling[s]), _mean(a.rolling[s])
            f[f"diff_{s}"] = f[f"h_{s}"] - f[f"a_{s}"]
        for p in self.spec["prematch"]:
            hv = pd.to_numeric(row.get(f"home_{p}", np.nan), errors="coerce")
            av = pd.to_numeric(row.get(f"away_{p}", np.nan), errors="coerce")
            f[f"home_{p}"], f[f"away_{p}"], f[f"diff_{p}"] = hv, av, hv - av
        return self.extra_features(f)

    # ── 결과 반영 (상태 변경) ─────────────────────────────────────────
    def update(self, row: dict):
        row = self.derive_match(dict(row))
        h, a = self.teams[row["home"]], self.teams[row["away"]]
        hs, as_ = float(row["home_score"]), float(row["away_score"])
        win, draw, loss = self.spec["points"]

        score_h = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        exp_h = 1 / (1 + 10 ** (-(h.elo + self.spec["elo_home_adv"] - a.elo) / 400))
        k = self.spec["elo_k"] * (1 + np.log1p(abs(hs - as_) / self.spec["mov_scale"]))
        h.elo += k * (score_h - exp_h)
        a.elo -= k * (score_h - exp_h)

        pts_h = win if hs > as_ else draw if hs == as_ else loss
        pts_a = win if as_ > hs else draw if hs == as_ else loss
        h.form.append((pts_h, hs, as_)); h.home_form.append((pts_h, hs, as_))
        a.form.append((pts_a, as_, hs)); a.away_form.append((pts_a, as_, hs))
        for s in self.spec["rolling_stats"]:
            for team, side in ((h, "home"), (a, "away")):
                v = pd.to_numeric(row.get(f"{side}_{s}", np.nan), errors="coerce")
                if np.isfinite(v):
                    team.rolling[s].append(float(v))
        h.last_date = a.last_date = row["date"]

    # ── 일괄 처리 ─────────────────────────────────────────────────────
    def build_history(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for rec in matches.sort_values("date").to_dict("records"):
            feats = self.features_for(rec)
            feats.update(date=rec["date"], league=rec["league"], home=rec["home"], away=rec["away"],
                         home_score=rec["home_score"], away_score=rec["away_score"])
            rows.append(feats)
            self.update(rec)
        return pd.DataFrame(rows)

    def build_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows, today = [], pd.Timestamp.today().normalize()
        for rec in fixtures.to_dict("records"):
            d = pd.to_datetime(rec.get("date", today), errors="coerce")
            rec["date"] = today if pd.isna(d) else d
            for t in (rec["home"], rec["away"]):
                if t not in self.teams:
                    log.warning("[%s] 학습 이력 없는 팀: %s (TEAM_ALIASES 확인)", self.sport, t)
            rows.append(self.features_for(rec))
        return pd.DataFrame(rows)


class SoccerFeatureBuilder(FeatureBuilder):
    """축구: 무승부 변수 - 전력 균형, 양팀 무승부 성향, 저득점 성향."""

    def derive_match(self, row):
        draw = float(row["home_score"] == row["away_score"])
        total = float(row["home_score"] + row["away_score"])
        row.update(home_draw=draw, away_draw=draw, home_total=total, away_total=total)
        return row

    def extra_feature_names(self):
        return ["draw_tendency", "low_scoring", "parity"]

    def extra_features(self, f):
        f["draw_tendency"] = _mean([f.get("h_draw"), f.get("a_draw")])
        f["low_scoring"] = _mean([f.get("h_total"), f.get("a_total")])
        f["parity"] = 1 / (1 + f["elo_abs_diff"] / 100)   # 전력 차 작을수록 1 → 무승부 확률↑
        return f


class BaseballFeatureBuilder(FeatureBuilder):
    """야구: 선발투수 매치업 - 선발 ERA/WHIP 차이 × 상대 타선 득점력."""

    def extra_feature_names(self):
        return ["sp_matchup_home", "sp_matchup_away"]

    def extra_features(self, f):
        # 홈 타선 득점력 × 원정 선발 실점력 (ERA 높을수록 홈 득점 기대↑)
        f["sp_matchup_home"] = f["h_all_gf"] * f["away_sp_era"] / 4.0 if np.isfinite(f["away_sp_era"]) else np.nan
        f["sp_matchup_away"] = f["a_all_gf"] * f["home_sp_era"] / 4.0 if np.isfinite(f["home_sp_era"]) else np.nan
        return f


class BasketballFeatureBuilder(FeatureBuilder):
    """농구: PACE × 레이팅 기대득점, 후반(3·4쿼터) 집중력, 백투백."""

    def extra_feature_names(self):
        return ["exp_pts_home", "exp_pts_away", "net_rtg_diff", "clutch_diff"]

    def extra_features(self, f):
        pace = _mean([f.get("h_pace"), f.get("a_pace")])
        f["exp_pts_home"] = pace * _mean([f.get("h_ortg"), f.get("a_drtg")]) / 100
        f["exp_pts_away"] = pace * _mean([f.get("a_ortg"), f.get("h_drtg")]) / 100
        f["net_rtg_diff"] = (f.get("h_ortg", np.nan) - f.get("h_drtg", np.nan)) - \
                            (f.get("a_ortg", np.nan) - f.get("a_drtg", np.nan))
        f["clutch_diff"] = f.get("h_q4", np.nan) - f.get("a_q4", np.nan)
        return f


class VolleyballFeatureBuilder(FeatureBuilder):
    """배구: 공격-리시브 효율 교차."""

    def extra_feature_names(self):
        return ["atk_vs_rcv"]

    def extra_features(self, f):
        f["atk_vs_rcv"] = (f.get("h_attack_eff", np.nan) - f.get("a_receive_eff", np.nan)) - \
                          (f.get("a_attack_eff", np.nan) - f.get("h_receive_eff", np.nan))
        return f


BUILDERS = {
    "soccer": SoccerFeatureBuilder, "baseball": BaseballFeatureBuilder,
    "basketball": BasketballFeatureBuilder, "volleyball": VolleyballFeatureBuilder,
}


# ══════════════════════════════════════════════════════════════════════
# 2. 앙상블 모델
# ══════════════════════════════════════════════════════════════════════
class EnsembleClassifier:
    """XGBoost + LightGBM 확률 가중 평균."""

    def __init__(self, n_classes: int, weights=config.ENSEMBLE_WEIGHTS):
        self.weights = weights
        if n_classes > 2:
            xo, lo = {"objective": "multi:softprob"}, {"objective": "multiclass"}
        else:
            xo, lo = {"objective": "binary:logistic"}, {"objective": "binary"}
        self.xgb = XGBClassifier(**config.XGB_PARAMS, **xo, random_state=config.RANDOM_STATE)
        self.lgbm = LGBMClassifier(**config.LGBM_PARAMS, **lo, random_state=config.RANDOM_STATE)

    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.lgbm.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        wx, wl = self.weights
        p = wx * self.xgb.predict_proba(X) + wl * self.lgbm.predict_proba(X)
        return p / p.sum(axis=1, keepdims=True)


class EnsembleRegressor:
    """홈/원정 예상 득점. 득점이 카운트(축구·야구)면 포아송 목적함수."""

    def __init__(self, count_target: bool, weights=config.ENSEMBLE_WEIGHTS):
        self.weights = weights
        xo = {"objective": "count:poisson"} if count_target else {"objective": "reg:squarederror"}
        lo = {"objective": "poisson"} if count_target else {"objective": "regression"}
        self.xgb = XGBRegressor(**config.XGB_PARAMS, **xo, random_state=config.RANDOM_STATE)
        self.lgbm = LGBMRegressor(**config.LGBM_PARAMS, **lo, random_state=config.RANDOM_STATE)

    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.lgbm.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        wx, wl = self.weights
        return np.clip(wx * self.xgb.predict(X) + wl * self.lgbm.predict(X), 0.01, None)


# ══════════════════════════════════════════════════════════════════════
# 3. 몬테카를로 시뮬레이터
# ══════════════════════════════════════════════════════════════════════
class SimResult:
    """N회 가상 경기 결과 (홈/원정 점수 배열) + 마켓별 확률 계산."""

    def __init__(self, home: np.ndarray, away: np.ndarray):
        self.home, self.away = home, away
        self.diff, self.total = home - away, home + away

    def wdl(self) -> dict:
        return {"H": float(np.mean(self.diff > 0)), "D": float(np.mean(self.diff == 0)),
                "A": float(np.mean(self.diff < 0))}

    def handicap(self, line: float) -> dict:
        """홈 기준 핸디캡 (예: -1.5). 정수 라인이면 D(핸디 무승부) 발생."""
        adj = self.diff + line
        return {"H": float(np.mean(adj > 0)), "D": float(np.mean(adj == 0)), "A": float(np.mean(adj < 0))}

    def over_under(self, line: float) -> dict:
        return {"O": float(np.mean(self.total > line)), "U": float(np.mean(self.total < line)),
                "P": float(np.mean(self.total == line))}

    def margin(self, m: int) -> list[float]:
        """승X패: 홈 m+1점차 이상 승 / |차이|<=m / 원정 m+1점차 이상 승."""
        return [float(np.mean(self.diff > m)), float(np.mean(np.abs(self.diff) <= m)),
                float(np.mean(self.diff < -m))]

    def score_bucket(self, side: str, buckets: list[tuple[int, int]]) -> list[float]:
        s = self.home if side == "home" else self.away
        return [float(np.mean((s >= lo) & (s <= hi))) for lo, hi in buckets]

    def set_score(self, labels: list[str]) -> list[float]:
        out = []
        for lab in labels:
            hs, as_ = map(int, lab.split(":"))
            out.append(float(np.mean((self.home == hs) & (self.away == as_))))
        return out


class MonteCarloSimulator:
    def __init__(self, sport: str, n_sims: int = config.MC_SIMULATIONS, seed: int = config.RANDOM_STATE):
        self.sport, self.spec, self.n = sport, config.SPORTS[sport], n_sims
        self.rng = np.random.default_rng(seed)

    def run(self, mu_home: float, mu_away: float, p_home: float | None = None,
            league: str | None = None) -> SimResult:
        dist = self.spec["score_dist"]
        if dist == "poisson":
            # TODO: Dixon-Coles / 이변량 포아송으로 저득점 무승부 보정
            h, a = self.rng.poisson(mu_home, self.n), self.rng.poisson(mu_away, self.n)
        elif dist == "negbin":
            h, a = self._negbin(mu_home), self._negbin(mu_away)
            h, a = self._extra_innings(h, a, mu_home, mu_away, league)
        elif dist == "normal":
            sd = self.spec["score_std"]
            h = np.rint(self.rng.normal(mu_home, sd, self.n)).astype(int)
            a = np.rint(self.rng.normal(mu_away, sd, self.n)).astype(int)
            h, a = self._overtime(h, a, mu_home, mu_away)
        elif dist == "sets":
            h, a = self._sets(p_home if p_home is not None else 0.5)
        else:
            raise ValueError(dist)
        return SimResult(h, a)

    def _negbin(self, mu: float) -> np.ndarray:
        r = self.spec["nb_dispersion"]
        return self.rng.negative_binomial(r, r / (r + mu), self.n)

    def _extra_innings(self, h, a, mu_h, mu_a, league):
        max_extra = self.spec["max_extra_innings"].get(league, 20)
        for _ in range(max_extra):
            tie = h == a
            if not tie.any():
                break
            h[tie] += self.rng.poisson(mu_h / 9, tie.sum())
            a[tie] += self.rng.poisson(mu_a / 9, tie.sum())
        return h, a   # KBO 는 연장 후에도 동점(무승부) 가능

    def _overtime(self, h, a, mu_h, mu_a):
        for _ in range(5):
            tie = h == a
            if not tie.any():
                break
            h[tie] += np.rint(self.rng.normal(mu_h * 5 / 48, 3.5, tie.sum())).astype(int)
            a[tie] += np.rint(self.rng.normal(mu_a * 5 / 48, 3.5, tie.sum())).astype(int)
        tie = h == a
        h[tie] += 1
        return h, a

    @staticmethod
    def set_prob_from_match(p_match: float) -> float:
        """5세트 3선승 매치 승률 → 세트 1개 승률 (이분법 역산)."""
        def match_p(p):
            q = 1 - p
            return p ** 3 * (1 + 3 * q + 6 * q ** 2)
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if match_p(mid) < p_match else (lo, mid)
        return (lo + hi) / 2

    def _sets(self, p_home: float):
        ps = self.set_prob_from_match(p_home)
        wins = self.rng.random((self.n, 5)) < ps
        h, a = np.zeros(self.n, int), np.zeros(self.n, int)
        for i in range(5):
            live = (h < 3) & (a < 3)
            h += live & wins[:, i]
            a += live & ~wins[:, i]
        return h, a


# ══════════════════════════════════════════════════════════════════════
# 4. 종목 파이프라인
# ══════════════════════════════════════════════════════════════════════
def _brier(y, p) -> float:
    return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))


def evaluate(y, p, name: str) -> dict:
    res = {"model": name, "n": len(y), "log_loss": log_loss(y, p, labels=list(range(p.shape[1]))),
           "brier": _brier(y, p), "accuracy": accuracy_score(y, p.argmax(axis=1))}
    log.info("[%s] n=%d logloss=%.4f brier=%.4f acc=%.3f", name, res["n"], res["log_loss"],
             res["brier"], res["accuracy"])
    return res


class SportModel:
    def __init__(self, sport: str):
        self.sport = sport
        self.spec = config.SPORTS[sport]
        self.outcomes = self.spec["outcomes"]
        self.builder: FeatureBuilder = BUILDERS.get(sport, FeatureBuilder)(sport)
        self.clf_odds: EnsembleClassifier | None = None
        self.clf_plain: EnsembleClassifier | None = None
        self.reg_home: EnsembleRegressor | None = None
        self.reg_away: EnsembleRegressor | None = None
        self.metrics: list[dict] = []

    # ── 라벨 ──────────────────────────────────────────────────────────
    def labels(self, data: pd.DataFrame) -> pd.Series:
        diff = data["home_score"] - data["away_score"]
        lab = np.where(diff > 0, "H", np.where(diff < 0, "A", "D"))
        idx = {o: i for i, o in enumerate(self.outcomes)}
        return pd.Series(lab, index=data.index).map(idx)   # 2-way 종목의 무승부는 NaN → 학습 제외

    # ── 학습 ──────────────────────────────────────────────────────────
    def _fit_eval(self, X, y, name, factory):
        split = int(len(X) * (1 - config.TEST_RATIO))
        m = factory().fit(X.iloc[:split], y[:split])
        if isinstance(m, EnsembleClassifier):
            self.metrics.append(evaluate(y[split:], m.predict_proba(X.iloc[split:]), name))
        else:
            mae = mean_absolute_error(y[split:], m.predict(X.iloc[split:]))
            self.metrics.append({"model": name, "n": len(y) - split, "mae": mae})
            log.info("[%s] MAE=%.3f", name, mae)
        return factory().fit(X, y)   # 최종 모델은 전체 데이터로 재학습

    def train(self, matches: pd.DataFrame) -> list[dict]:
        self.builder = BUILDERS.get(self.sport, FeatureBuilder)(self.sport)
        self.metrics = []
        data = self.builder.build_history(matches)
        data = data.dropna(subset=["h_all_pts", "a_all_pts"]).reset_index(drop=True)
        if len(data) < config.MIN_TRAIN_ROWS:
            log.warning("[%s] 학습 데이터 %d행 - 너무 적음", self.sport, len(data))

        y = self.labels(data)
        ok = y.notna().to_numpy()
        n_cls = len(self.outcomes)
        plain_cols = self.builder.feature_columns(with_odds=False)
        odds_cols = self.builder.feature_columns(with_odds=True)
        imp_cols = [f"imp_{o}" for o in self.outcomes]

        # (1) 배당 포함 분류기 + 북메이커 기준선
        has_odds = ok & data[imp_cols].notna().all(axis=1).to_numpy()
        if has_odds.sum() >= config.MIN_TRAIN_ROWS:
            Xo = data.loc[has_odds, odds_cols].reset_index(drop=True)
            yo = y[has_odds].astype(int).to_numpy()
            self.clf_odds = self._fit_eval(Xo, yo, f"{self.sport}:clf+odds", lambda: EnsembleClassifier(n_cls))
            split = int(len(Xo) * (1 - config.TEST_RATIO))
            self.metrics.append(evaluate(yo[split:], Xo.iloc[split:][imp_cols].to_numpy(), f"{self.sport}:bookmaker"))
        else:
            log.info("[%s] 배당 데이터 부족 → 배당 포함 분류기 생략", self.sport)

        # (2) Elo/폼 전용 분류기
        self.clf_plain = self._fit_eval(data.loc[ok, plain_cols].reset_index(drop=True),
                                        y[ok].astype(int).to_numpy(), f"{self.sport}:clf",
                                        lambda: EnsembleClassifier(n_cls))

        # (3) 득점 회귀 (몬테카를로 입력)
        count = self.spec["score_dist"] in ("poisson", "negbin")
        X = data[plain_cols]
        self.reg_home = self._fit_eval(X, data["home_score"].to_numpy(), f"{self.sport}:reg_home",
                                       lambda: EnsembleRegressor(count))
        self.reg_away = self._fit_eval(X, data["away_score"].to_numpy(), f"{self.sport}:reg_away",
                                       lambda: EnsembleRegressor(count))
        return self.metrics

    # ── 예측 ──────────────────────────────────────────────────────────
    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """fixtures(home, away, [date], [odds_*], [home_<prematch>…]) → p_H/p_D/p_A, mu_home, mu_away."""
        fixtures = apply_aliases(fixtures).reset_index(drop=True)
        X = self.builder.build_fixtures(fixtures)
        imp_cols = [f"imp_{o}" for o in self.outcomes]
        use_odds = X[imp_cols].notna().all(axis=1).to_numpy() & (self.clf_odds is not None)

        probs = np.zeros((len(X), len(self.outcomes)))
        if use_odds.any():
            probs[use_odds] = self.clf_odds.predict_proba(X.loc[use_odds, self.builder.feature_columns(True)])
        if (~use_odds).any():
            probs[~use_odds] = self.clf_plain.predict_proba(X.loc[~use_odds, self.builder.feature_columns(False)])

        out = fixtures.copy()
        for i, o in enumerate(self.outcomes):
            out[f"p_{o}"] = probs[:, i]
        if "D" not in self.outcomes:
            out["p_D"] = 0.0
        plain = X[self.builder.feature_columns(False)]
        out["mu_home"], out["mu_away"] = self.reg_home.predict(plain), self.reg_away.predict(plain)
        out["model_used"] = np.where(use_odds, "odds", "no-odds")
        return out

    def simulate(self, row: pd.Series | dict, n_sims: int = config.MC_SIMULATIONS) -> SimResult:
        return MonteCarloSimulator(self.sport, n_sims).run(
            row["mu_home"], row["mu_away"], p_home=row.get("p_H"), league=row.get("league"))

    # ── 저장 / 로드 ───────────────────────────────────────────────────
    def save(self):
        joblib.dump(self, config.model_file(self.sport))
        log.info("[%s] 모델 저장 → %s", self.sport, config.model_file(self.sport))

    @staticmethod
    def load(sport: str) -> "SportModel":
        path = config.model_file(sport)
        if not path.exists():
            raise FileNotFoundError(f"{path.name} 없음 - 먼저 `python main.py train --sport {sport}` 실행")
        return joblib.load(path)


class ModelFactory:
    @staticmethod
    def create(sport: str) -> SportModel:
        if sport not in config.SPORTS:
            raise ValueError(f"지원하지 않는 종목: {sport}")
        return SportModel(sport)

    @staticmethod
    def load(sport: str) -> SportModel:
        return SportModel.load(sport)


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("home", "away"):
        df[col] = df[col].astype(str).str.strip().replace(config.TEAM_ALIASES)
    return df


def train(sport: str) -> list[dict]:
    from scraper import load_matches
    model = ModelFactory.create(sport)
    metrics = model.train(load_matches(sport))
    model.save()
    return metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(sys.argv[1] if len(sys.argv) > 1 else "soccer")
