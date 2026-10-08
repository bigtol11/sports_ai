"""
main.py
-------
통합 실행 진입점 (CLI).

  python main.py scrape --sport soccer            # 데이터 수집 (soccer/baseball/basketball/volleyball/all)
  python main.py scrape --sport basketball --demo # 스크래퍼 미구현 종목: 가상 데이터로 파이프라인 검증
  python main.py train  --sport soccer            # 종목별 모델 학습 + 검증 지표
  python main.py proto  [--file 경로] [--bankroll 100000]
  python main.py toto   --type soccer_wdl [--file 경로] [--budget 9600]
  python main.py check                            # 파일/모듈/모델 상태 점검 + 완료 메시지
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

import config
import model
import proto_analyser
import scraper
import toto_scheduler

DEFAULT_PROTO_FILE = config.INPUT_DIR / "proto_odds.csv"


def _sports(arg: str) -> list[str]:
    return list(config.SPORTS) if arg == "all" else [arg]


def cmd_scrape(args):
    for sport in _sports(args.sport):
        scraper.collect(sport, demo=args.demo)


def cmd_train(args):
    for sport in _sports(args.sport):
        metrics = model.train(sport)
        print(pd.DataFrame(metrics).to_string(index=False, float_format="%.4f"))


def cmd_proto(args):
    _singles, combos, portfolio = proto_analyser.run(pd.read_csv(args.file or DEFAULT_PROTO_FILE),
                                                     bankroll=args.bankroll)
    print("\n[추천 2폴더 조합 TOP 5]")
    print(combos.head(5).to_string(index=False, float_format="%.3f") if not combos.empty else "없음")
    print("\n[최종 베팅 포트폴리오]")
    print(portfolio.to_string(index=False, float_format="%.3f") if not portfolio.empty else "가치 베팅 없음 (패스 권장)")


def cmd_toto(args):
    path = args.file or config.INPUT_DIR / f"toto_{args.type}.csv"
    _report, _tickets, _summary, slip = toto_scheduler.run(args.type, pd.read_csv(path), budget=args.budget)
    print(slip)


REQUIRED_FILES = ["config.py", "scraper.py", "model.py", "proto_analyser.py", "toto_scheduler.py",
                  "main.py", "README.md", "AGENTS.md", "CLAUDE.md", "requirements.txt"]


def cmd_check(_args):
    """에이전트 작업 완료 확인용: 필수 파일 · 모듈 import · 종목별 데이터/모델 상태."""
    import importlib
    ok = True
    print("[파일]")
    for name in REQUIRED_FILES:
        exists = (config.BASE_DIR / name).exists()
        ok &= exists
        print(f"  {'✅' if exists else '❌'} {name}")
    print("[모듈]")
    for mod in ("bs4", "selenium", "xgboost", "lightgbm", "sklearn", "pandas"):
        try:
            importlib.import_module(mod)
            print(f"  ✅ {mod}")
        except ImportError:
            ok = False
            print(f"  ❌ {mod} (pip install -r requirements.txt)")
    print("[종목 상태]")
    for sport in config.SPORTS:
        data, mdl = config.matches_file(sport), config.model_file(sport)
        n = len(pd.read_csv(data, usecols=["league"])) if data.exists() else 0
        demo = data.exists() and (pd.read_csv(data, usecols=["league"])["league"] == "DEMO").any()
        print(f"  {sport:<11} 데이터 {n:>6}경기{' (가상)' if demo else ''} | 모델 {'✅' if mdl.exists() else '⬜'}")
    print()
    print("=" * 60)
    print("✅ 완료: 초고차원 진화형 스포츠 예측 AI 시스템 뼈대 + 마스터 설계도(README.md) 준비 완료"
          if ok else "❌ 점검 실패: 위 ❌ 항목을 확인하세요")
    print("   다음 단계 → README.md §2 Codex 즉시 실행 지침의 작업 큐 T1부터 진행")
    print("=" * 60)
    raise SystemExit(0 if ok else 1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="초고차원 진화형 스포츠 예측 AI 시스템")
    sub = parser.add_subparsers(dest="command", required=True)
    sports = list(config.SPORTS) + ["all"]

    p = sub.add_parser("scrape", help="경기 데이터 수집")
    p.add_argument("--sport", choices=sports, default="soccer")
    p.add_argument("--demo", action="store_true", help="가상 데이터 생성 (파이프라인 검증용)")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("train", help="모델 학습")
    p.add_argument("--sport", choices=sports, default="soccer")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("proto", help="프로토 EV/켈리 분석")
    p.add_argument("--file")
    p.add_argument("--bankroll", type=float, default=config.BANKROLL)
    p.set_defaults(func=cmd_proto)

    p = sub.add_parser("toto", help="토토 회차 최적 마킹")
    p.add_argument("--type", choices=list(config.TOTO_GAME_TYPES), default="soccer_wdl")
    p.add_argument("--file")
    p.add_argument("--budget", type=int, default=config.TOTO_DEFAULT_BUDGET)
    p.set_defaults(func=cmd_toto)

    sub.add_parser("check", help="상태 점검 + 완료 메시지").set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
