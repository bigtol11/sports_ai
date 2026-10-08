"""
toto_scheduler.py
-----------------
[역할] 토토 회차(축구 승무패 / 야구 승1패 / 농구 승5패 / 스페셜 / 배구 매치 등)의 지정 경기를 받아
       예산(조합 수) 안에서 '전 경기 적중 확률'을 최대화하는 복식 마킹을 설계하고 슬립지 레이아웃을 출력한다.

입력 CSV:
    game, date, league, home, away [, pub_1, pub_2, ... ]   (pub_i = i번째 선택지 대중 투표율 %, 선택)
게임 유형: config.TOTO_GAME_TYPES  (rule: wdl / margin / score_bucket / set_score)

최적화 (전역 최적):
    라인 i 에서 확률 상위 k_i 개 선택지를 마킹 (k_i = 1 단일, 2 복수, 3 이상 다중)
    조합 수 C = Π k_i ≤ 예산 / 조합당 금액
    적중 확률 P = Π (라인 i 의 상위 k_i 확률 합)
    → 상태 = 누적 조합 수 C 인 동적계획법으로 log P 최대화.

출력:
    output/toto_<type>_marking.csv   라인별 마킹 + 확률 + 역배 가치
    output/toto_<type>_tickets.csv   실제 구매할 전체 조합
    output/toto_<type>_slip.txt      슬립지 형태 레이아웃
"""
from __future__ import annotations

import itertools
import logging
import math
import unicodedata

import numpy as np
import pandas as pd

import config
from model import ModelFactory

log = logging.getLogger(__name__)

MAX_TICKET_ROWS = 20_000   # 티켓 전개 상한 (그 이상은 CSV 생략)


# ══════════════════════════════════════════════════════════════════════
# 1. 라인별 확률 생성
# ══════════════════════════════════════════════════════════════════════
def build_lines(game_type: str, fixtures: pd.DataFrame) -> pd.DataFrame:
    """회차 경기 → 마킹 라인 목록. 스페셜(score_bucket)은 경기당 홈/원정 2라인."""
    gt = config.TOTO_GAME_TYPES[game_type]
    model = ModelFactory.load(gt["sport"])
    pred = model.predict(fixtures)
    labels = gt["labels"]
    rows = []
    for i, r in enumerate(pred.to_dict("records")):
        base = {"game": r.get("game", i + 1), "match": f"{r['home']} vs {r['away']}"}
        pub = [r.get(f"pub_{j + 1}", np.nan) for j in range(len(labels))]
        if gt["rule"] == "wdl":
            rows.append({**base, "line": "", "probs": [r["p_H"], r["p_D"], r["p_A"]], "public": pub})
            continue
        sim = model.simulate(r)
        if gt["rule"] == "margin":
            rows.append({**base, "line": "", "probs": sim.margin(gt["margin"]), "public": pub})
        elif gt["rule"] == "score_bucket":
            for side, name in (("home", "홈 득점"), ("away", "원정 득점")):
                rows.append({**base, "line": name, "probs": sim.score_bucket(side, gt["buckets"]),
                             "public": [np.nan] * len(labels)})
        elif gt["rule"] == "set_score":
            rows.append({**base, "line": "", "probs": sim.set_score(labels), "public": pub})
        else:
            raise ValueError(gt["rule"])
    lines = pd.DataFrame(rows)
    lines["probs"] = lines["probs"].apply(lambda p: list(np.asarray(p) / max(sum(p), 1e-12)))
    return lines


# ══════════════════════════════════════════════════════════════════════
# 2. 최적 복식 마킹 (DP)
# ══════════════════════════════════════════════════════════════════════
def optimise_marking(lines: pd.DataFrame, max_combinations: int) -> list[list[int]]:
    """라인별로 마킹할 선택지 인덱스 리스트를 반환."""
    options = []
    for probs in lines["probs"]:
        order = list(np.argsort(probs)[::-1])
        opts = {}
        for k in range(1, len(probs) + 1):
            cover = float(sum(probs[j] for j in order[:k]))
            opts[k] = (math.log(max(cover, 1e-12)), sorted(order[:k]))
        options.append(opts)

    dp: dict[int, tuple[float, list]] = {1: (0.0, [])}   # 누적 조합 수 → (log P, 마킹)
    for opts in options:
        nxt: dict[int, tuple[float, list]] = {}
        for combos, (lp, marks) in dp.items():
            for k, (gain, mark) in opts.items():
                c = combos * k
                if c > max_combinations:
                    continue
                cand = (lp + gain, marks + [mark])
                if c not in nxt or cand[0] > nxt[c][0]:
                    nxt[c] = cand
        dp = nxt
    return max(dp.values(), key=lambda x: x[0])[1]


# ══════════════════════════════════════════════════════════════════════
# 3. 리포트 / 티켓 / 슬립지
# ══════════════════════════════════════════════════════════════════════
def cover_type(k: int, n_labels: int) -> str:
    return "올커버" if k == n_labels else {1: "단일", 2: "복수"}.get(k, f"{k}중")


def build_report(lines: pd.DataFrame, marks: list[list[int]], labels: list[str]) -> pd.DataFrame:
    rows = []
    for no, (r, mark) in enumerate(zip(lines.to_dict("records"), marks), start=1):
        probs, pub = r["probs"], r["public"]
        row = {"no": no, "game": r["game"], "match": r["match"], "line": r["line"]}
        row.update({lab: round(p, 3) for lab, p in zip(labels, probs)})
        row.update({
            "marking": "+".join(labels[j] for j in mark),
            "type": cover_type(len(mark), len(labels)),
            "cover_prob": round(sum(probs[j] for j in mark), 4),
        })
        if all(np.isfinite(pub)):
            # 모델 확률 / 대중 투표율 > 1 → 대중이 과소평가 → 적중 시 배당 유리 (역배 가치)
            ratio = [p / max(v / 100, 1e-6) for p, v in zip(probs, pub)]
            j = int(np.argmax(ratio))
            row.update(value_pick=labels[j], value_ratio=round(ratio[j], 2),
                       upset_alert=int(np.argmax(probs)) != int(np.argmax(pub)))
        rows.append(row)
    return pd.DataFrame(rows)


def expand_tickets(marks: list[list[int]], labels: list[str]) -> pd.DataFrame | None:
    n = math.prod(len(m) for m in marks)
    if n > MAX_TICKET_ROWS:
        log.warning("조합 %d개 - 티켓 CSV 전개 생략", n)
        return None
    return pd.DataFrame([[labels[j] for j in t] for t in itertools.product(*marks)],
                        columns=[f"L{i}" for i in range(1, len(marks) + 1)])


def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    while _w(s) > width:
        s = s[:-1]
    return s + " " * (width - _w(s))


def render_slip(report: pd.DataFrame, marks: list[list[int]], labels: list[str],
                title: str, summary: dict) -> str:
    """슬립지 레이아웃: ■ = 마킹, □ = 미마킹."""
    mw, cw = 48, max(4, max(_w(l) for l in labels) + 2)
    sep = "+" + "-" * 4 + "+" + "-" * (mw + 2) + "+" + "+".join("-" * cw for _ in labels) + "+"
    out = [f"[ {title} ]", sep,
           "|" + _pad(" No", 4) + "| " + _pad("경기", mw) + " |" + "|".join(_pad(" " + l, cw) for l in labels) + "|",
           sep]
    for r, mark in zip(report.to_dict("records"), marks):
        name = r["match"] + (f" · {r['line']}" if r["line"] else "")
        cells = "|".join(_pad(" ■" if j in mark else " □", cw) for j in range(len(labels)))
        out.append("|" + _pad(f" {r['no']:02d}", 4) + "| " + _pad(name, mw) + " |" + cells + "|")
    out += [sep,
            f"마킹 분배: " + " / ".join(f"{k} {v}" for k, v in summary["distribution"].items()),
            f"조합 {summary['combinations']}개 × {config.TOTO_COST_PER_COMBINATION}원 = {summary['cost_krw']:,}원",
            f"전 라인 적중 확률(모델 기준): {summary['all_hit_prob'] * 100:.4f}%"]
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════
# 4. 실행
# ══════════════════════════════════════════════════════════════════════
def run(game_type: str, fixtures: pd.DataFrame, budget: int = config.TOTO_DEFAULT_BUDGET):
    gt = config.TOTO_GAME_TYPES[game_type]
    labels = gt["labels"]
    if gt["n_games"] and len(fixtures) != gt["n_games"]:
        log.warning("%s 는 %d경기여야 합니다 (입력 %d경기).", game_type, gt["n_games"], len(fixtures))
    max_comb = max(1, budget // config.TOTO_COST_PER_COMBINATION)

    lines = build_lines(game_type, fixtures)
    marks = optimise_marking(lines, max_comb)
    report = build_report(lines, marks, labels)
    tickets = expand_tickets(marks, labels)

    n_comb = math.prod(len(m) for m in marks)
    summary = {
        "combinations": n_comb,
        "cost_krw": n_comb * config.TOTO_COST_PER_COMBINATION,
        "all_hit_prob": float(np.prod(report["cover_prob"])),
        "distribution": report["type"].value_counts().to_dict(),
    }
    slip = render_slip(report, marks, labels, f"{game_type} 마킹 슬립", summary)

    prefix = config.OUTPUT_DIR / f"toto_{game_type}"
    report.to_csv(f"{prefix}_marking.csv", index=False, encoding="utf-8-sig")
    if tickets is not None:
        tickets.to_csv(f"{prefix}_tickets.csv", index=False, encoding="utf-8-sig")
    (config.OUTPUT_DIR / f"toto_{game_type}_slip.txt").write_text(slip, encoding="utf-8")
    log.info("[%s] %d조합 / %s원 / 전체 적중확률 %.4f%%", game_type, n_comb,
             f"{summary['cost_krw']:,}", summary["all_hit_prob"] * 100)
    return report, tickets, summary, slip
