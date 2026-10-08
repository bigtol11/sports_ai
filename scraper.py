"""
scraper.py
----------
[역할] 전 종목(축구·야구·농구·배구) 경기 결과/스탯/배당 수집 → 종목별 표준 스키마로 정규화·저장.

구조 (팩토리 패턴):
    ScraperFactory.create("soccer")  → SoccerScraper
    ScraperFactory.create("baseball")→ BaseballScraper
    ...
  각 Scraper 는 리그 코드별 fetcher 메서드를 가지며, collect() 가 리그×시즌을 돌며 수집한다.

구현 현황:
  ✅ 축구  : football-data.co.uk (EPL/5대리그 등, 슈팅·코너·배당 포함)        - 실동작
  ✅ 야구  : MLB Stats API (statsapi.mlb.com, 결과 + 예고 선발)               - 실동작
  ⬜ 야구  : KBO / 농구 NBA·KBL / 배구 V리그 / 축구 K리그                     - 스켈레톤 (TODO 표시)
  🧪 전 종목: generate_synthetic() - 파이프라인 검증용 가상 데이터 (실전 사용 금지)

표준 스키마 (모든 종목 공통):
    date, sport, league, season, home, away, home_score, away_score,
    odds_home, odds_draw, odds_away,
    home_<stat>, away_<stat>   ← config.SPORTS[sport]["rolling_stats"] / ["prematch"]
"""
from __future__ import annotations

import io
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

BASE_COLUMNS = [
    "date", "sport", "league", "season", "home", "away",
    "home_score", "away_score", "odds_home", "odds_draw", "odds_away",
]


def stat_columns(sport: str) -> list[str]:
    spec = config.SPORTS[sport]
    names = spec["rolling_stats"] + spec["prematch"]
    return [f"{side}_{n}" for n in names for side in ("home", "away")]


# ══════════════════════════════════════════════════════════════════════
# 기초 클래스
# ══════════════════════════════════════════════════════════════════════
class BaseScraper(ABC):
    sport: str = ""

    def __init__(self, leagues: list[str] | None = None, seasons: list[str] | None = None,
                 use_cache: bool = True):
        spec = config.SPORTS[self.sport]
        self.leagues = leagues or list(spec["leagues"])
        self.seasons = seasons or spec["seasons"]
        self.use_cache = use_cache
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT

    # 리그 코드 → 수집 함수. 하위 클래스에서 정의.
    @property
    @abstractmethod
    def fetchers(self) -> dict[str, Callable[[str], pd.DataFrame]]:
        ...

    # ── 공통 HTTP + 캐시 ──────────────────────────────────────────────
    def http_get(self, url: str, cache_name: str | None = None, refresh: bool = False,
                 **kwargs) -> str | None:
        cache = config.RAW_DIR / self.sport / cache_name if cache_name else None
        if cache and self.use_cache and cache.exists() and not refresh:
            return cache.read_text(encoding="utf-8", errors="ignore")
        try:
            resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("[%s] 요청 실패 %s: %s", self.sport, url, e)
            return None
        finally:
            time.sleep(config.REQUEST_DELAY)
        text = resp.content.decode("utf-8", errors="ignore")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        return text

    # ── 크롤링 헬퍼 (README §2.4 크롤러 구현 절차) ──────────────────────
    def get_json(self, url: str, cache_name: str | None = None, refresh: bool = False, **kwargs):
        """1순위: 사이트가 내부적으로 호출하는 JSON(XHR) 엔드포인트."""
        import json
        text = self.http_get(url, cache_name, refresh, **kwargs)
        return json.loads(text) if text else None

    def post_json(self, url: str, data: dict, headers: dict | None = None):
        """ASP.NET 웹서비스(KBO 등)처럼 POST 로 데이터를 주는 XHR 엔드포인트."""
        try:
            resp = self.session.post(url, data=data, headers=headers or {}, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("[%s] POST 실패 %s: %s", self.sport, url, e)
            return None
        finally:
            time.sleep(config.REQUEST_DELAY)

    def get_soup(self, url: str, cache_name: str | None = None, refresh: bool = False):
        """2순위: 서버 렌더링 HTML → BeautifulSoup(lxml)."""
        from bs4 import BeautifulSoup
        text = self.http_get(url, cache_name, refresh)
        return BeautifulSoup(text, "lxml") if text else None

    @staticmethod
    def browser(headless: bool = True):
        """3순위(최후 수단): JS 렌더링 전용 페이지 → Selenium Chrome. 사용 후 driver.quit() 필수."""
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        opts = webdriver.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument(f"user-agent={config.USER_AGENT}")
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

    def is_current(self, season: str) -> bool:
        """진행 중 시즌은 캐시를 쓰지 않고 매번 갱신."""
        return season == self.seasons[-1]

    # ── 정규화 ────────────────────────────────────────────────────────
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sport"] = self.sport
        for col in BASE_COLUMNS + stat_columns(self.sport):
            if col not in df.columns:
                df[col] = np.nan
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home", "away", "home_score", "away_score"])
        return df[BASE_COLUMNS + stat_columns(self.sport)]

    # ── 수집 실행 ─────────────────────────────────────────────────────
    def collect(self) -> pd.DataFrame:
        frames = []
        for league in self.leagues:
            fetch = self.fetchers.get(league)
            if fetch is None:
                log.warning("[%s] %s: 등록된 fetcher 없음", self.sport, league)
                continue
            for season in self.seasons:
                try:
                    df = fetch(league, season)
                except NotImplementedError as e:
                    log.warning("[%s] %s 미구현 - %s", self.sport, league, e)
                    break
                if df is not None and not df.empty:
                    df["league"], df["season"] = league, season
                    frames.append(self.standardize(df))
                    log.info("[%s] %s %s: %d경기", self.sport, league, season, len(df))
        if not frames:
            log.warning("[%s] 수집된 데이터 없음", self.sport)
            return pd.DataFrame(columns=BASE_COLUMNS + stat_columns(self.sport))
        return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# 축구
# ══════════════════════════════════════════════════════════════════════
class SoccerScraper(BaseScraper):
    sport = "soccer"
    FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    # 배당 우선순위: 피나클 종가 → 피나클 → 시장평균 → Bet365
    ODDS_PRIORITY = [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"),
                     ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]
    STAT_MAP = {"HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
                "HC": "home_corners", "AC": "away_corners"}

    @property
    def fetchers(self):
        fd = {code: self.fetch_football_data for code in config.SPORTS["soccer"]["leagues"] if code != "KLEAGUE"}
        return {**fd, "KLEAGUE": self.fetch_kleague}

    def fetch_football_data(self, league: str, season: str) -> pd.DataFrame | None:
        text = self.http_get(self.FD_URL.format(season=season, league=league),
                             cache_name=f"{league}_{season}.csv", refresh=self.is_current(season))
        if not text:
            return None
        raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
            "home": raw["HomeTeam"].str.strip(), "away": raw["AwayTeam"].str.strip(),
            "home_score": raw["FTHG"].astype(int), "away_score": raw["FTAG"].astype(int),
        })
        for src, dst in self.STAT_MAP.items():
            if src in raw.columns:
                df[dst] = pd.to_numeric(raw[src], errors="coerce")
        for col in ("odds_home", "odds_draw", "odds_away"):
            df[col] = np.nan
        for h, d, a in self.ODDS_PRIORITY:
            if {h, d, a}.issubset(raw.columns):
                m = df["odds_home"].isna()
                df.loc[m, "odds_home"] = pd.to_numeric(raw.loc[m, h], errors="coerce")
                df.loc[m, "odds_draw"] = pd.to_numeric(raw.loc[m, d], errors="coerce")
                df.loc[m, "odds_away"] = pd.to_numeric(raw.loc[m, a], errors="coerce")
        # TODO(xG): Understat JSON 병합 (README §2.5-A, 2026-10-09 구조 확인됨)
        #   GET https://understat.com/getLeagueData/{EPL|La_liga|Bundesliga|Serie_A|Ligue_1}/{시작연도}
        #   헤더 X-Requested-With: XMLHttpRequest, Referer: https://understat.com/league/EPL/{연도}
        #   → {"dates": [{h.title, a.title, goals.h/a, xG.h/a, datetime, isResult}]}
        #   팀명이 football-data 와 다름 → config.TEAM_ALIASES 류의 매핑 테이블 필요
        return df

    def fetch_kleague(self, league: str, season: str) -> pd.DataFrame:
        # TODO: K리그 데이터포털(data.kleague.com) 경기 결과 크롤링
        raise NotImplementedError("K리그 수집기 미구현 (data.kleague.com)")


# ══════════════════════════════════════════════════════════════════════
# 야구
# ══════════════════════════════════════════════════════════════════════
class BaseballScraper(BaseScraper):
    sport = "baseball"
    MLB_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}"
               "&gameType=R&hydrate=probablePitcher")

    @property
    def fetchers(self):
        return {"MLB": self.fetch_mlb, "KBO": self.fetch_kbo}

    def fetch_mlb(self, league: str, season: str) -> pd.DataFrame | None:
        import json
        text = self.http_get(self.MLB_URL.format(season=season), cache_name=f"MLB_{season}.json",
                             refresh=self.is_current(season))
        if not text:
            return None
        rows = []
        for day in json.loads(text).get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                h, a = g["teams"]["home"], g["teams"]["away"]
                if "score" not in h or "score" not in a:
                    continue
                rows.append({
                    "date": pd.to_datetime(g["gameDate"]).tz_convert(None).normalize(),
                    "home": h["team"]["name"], "away": a["team"]["name"],
                    "home_score": h["score"], "away_score": a["score"],
                    "home_sp_id": h.get("probablePitcher", {}).get("id"),
                    "away_sp_id": a.get("probablePitcher", {}).get("id"),
                })
        # TODO(선발투수): sp_id 로 /api/v1/people/{id}/stats?stats=gameLog&group=pitching&season={season} 조회 →
        #   '해당 경기 이전까지' 누적 ERA/WHIP/FIP/K9 를 계산해 home_sp_era 등에 채운다.
        #   (시즌 최종 ERA 를 쓰면 미래 정보 누수)
        # TODO(배당): MLB 배당은 별도 소스(OddsScraper) 병합
        return pd.DataFrame(rows)

    def fetch_kbo(self, league: str, season: str) -> pd.DataFrame:
        # TODO(KBO): README §2.5-B (2026-10-09 구조 확인됨)
        #   POST https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList
        #   data: leId=1, srIdList=0,9,6, seasonId=2025, gameMonth=07, teamId=
        #   헤더: X-Requested-With: XMLHttpRequest, Referer: https://www.koreabaseball.com/Schedule/Schedule.aspx
        #   → rows[].row[] 셀: Class=day("07.01(화)") / time / play(HTML: 팀명 + <span class="win|lose">점수</span>)
        #     relay 셀 링크의 gameId(예: 20250701SSOB0) → GameCenter 에서 선발투수 조회
        #   day 셀은 RowSpan 으로 생략되므로 직전 날짜를 이어받아야 함
        raise NotImplementedError("KBO 수집기 미구현 (koreabaseball.com)")


# ══════════════════════════════════════════════════════════════════════
# 농구
# ══════════════════════════════════════════════════════════════════════
class BasketballScraper(BaseScraper):
    sport = "basketball"

    @property
    def fetchers(self):
        return {"NBA": self.fetch_nba, "KBL": self.fetch_kbl}

    def fetch_nba(self, league: str, season: str) -> pd.DataFrame:
        # TODO: `pip install nba_api` 후
        #   leaguegamefinder.LeagueGameFinder(season_nullable=season) → 경기 결과
        #   boxscoreadvancedv2 → PACE / OFF_RATING / DEF_RATING
        #   boxscoresummaryv2 (LineScore) → 쿼터별 득점 q1~q4
        raise NotImplementedError("NBA 수집기 미구현 (nba_api 사용 권장)")

    def fetch_kbl(self, league: str, season: str) -> pd.DataFrame:
        # TODO: KBL 공식 사이트(kbl.or.kr) 경기 기록 크롤링
        raise NotImplementedError("KBL 수집기 미구현 (kbl.or.kr)")


# ══════════════════════════════════════════════════════════════════════
# 배구
# ══════════════════════════════════════════════════════════════════════
class VolleyballScraper(BaseScraper):
    sport = "volleyball"

    @property
    def fetchers(self):
        return {"VM": self.fetch_vleague, "VW": self.fetch_vleague}

    def fetch_vleague(self, league: str, season: str) -> pd.DataFrame:
        # TODO: KOVO 공식 사이트(kovo.co.kr) 경기 결과 크롤링
        #   home_score/away_score = 세트 스코어(3:1 등), 세트별 점수 → points_total
        #   공격성공률 attack_eff, 리시브효율 receive_eff, 블로킹 blocks, 서브에이스 aces
        raise NotImplementedError("V리그 수집기 미구현 (kovo.co.kr)")


# ══════════════════════════════════════════════════════════════════════
# 팩토리
# ══════════════════════════════════════════════════════════════════════
class ScraperFactory:
    _registry: dict[str, type[BaseScraper]] = {
        "soccer": SoccerScraper,
        "baseball": BaseballScraper,
        "basketball": BasketballScraper,
        "volleyball": VolleyballScraper,
    }

    @classmethod
    def register(cls, sport: str, scraper_cls: type[BaseScraper]):
        cls._registry[sport] = scraper_cls

    @classmethod
    def create(cls, sport: str, **kwargs) -> BaseScraper:
        if sport not in cls._registry:
            raise ValueError(f"지원하지 않는 종목: {sport} (가능: {list(cls._registry)})")
        return cls._registry[sport](**kwargs)


# ══════════════════════════════════════════════════════════════════════
# 배당 / 얼터너티브 데이터 (스켈레톤)
# ══════════════════════════════════════════════════════════════════════
class OddsScraper:
    """해외 배당 흐름(오픈→종가 변동) 수집. README §6-③ 참고."""

    def fetch_snapshots(self, sport: str, date: str) -> pd.DataFrame:
        # TODO: The Odds API(유료 키) 등에서 경기별 배당을 일정 간격으로 스냅샷 →
        #   columns: match_key, timestamp, book, odds_home, odds_draw, odds_away
        #   피처: 오픈 대비 종가 변동률, 스팀무브 여부, 북메이커 간 분산
        raise NotImplementedError


def load_betman_odds(path) -> pd.DataFrame:
    """베트맨 프로토 배당은 사용자가 CSV로 입력한다 (자동 크롤링은 이용약관 확인 후에만)."""
    return pd.read_csv(path)


class SentimentCollector:
    """뉴스/SNS 텍스트 수집 → 감성 점수. README §6-③ 참고."""

    def collect(self, team: str, since: str) -> pd.DataFrame:
        # TODO: 뉴스 RSS / 커뮤니티 게시글 수집 → (team, timestamp, text, source)
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
# 검증용 가상 데이터
# ══════════════════════════════════════════════════════════════════════
def generate_synthetic(sport: str, n_teams: int = 12, n_seasons: int = 6, seed: int = 0) -> pd.DataFrame:
    """
    잠재 전력값으로 그럴듯한 경기를 생성한다. 실전 예측에 쓰면 안 되며,
    스크래퍼 미구현 종목의 파이프라인(학습→예측→프로토→토토)이 동작하는지 확인하는 용도.
    """
    rng = np.random.default_rng(seed)
    spec = config.SPORTS[sport]
    teams = [f"{sport[:3].upper()}_Team{i:02d}" for i in range(n_teams)]
    strength = dict(zip(teams, rng.normal(0, 1, n_teams)))
    rows, date = [], pd.Timestamp("2023-01-01")
    for season in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                date += pd.Timedelta(hours=int(rng.integers(6, 30)))
                d = strength[h] - strength[a] + 0.25
                row = {"date": date, "league": "DEMO", "season": str(2023 + season), "home": h, "away": a}
                if sport == "soccer":
                    hs, as_ = rng.poisson(np.exp(0.3 + 0.3 * d)), rng.poisson(np.exp(0.1 - 0.3 * d))
                elif sport == "baseball":
                    hs, as_ = rng.poisson(4.5 + 0.8 * d), rng.poisson(4.3 - 0.8 * d)
                    row.update(home_sp_era=4.0 - 0.5 * strength[h] + rng.normal(0, .5),
                               away_sp_era=4.0 - 0.5 * strength[a] + rng.normal(0, .5))
                elif sport == "basketball":
                    hs, as_ = (int(rng.normal(112 + 4 * d, 11)), int(rng.normal(109 - 4 * d, 11)))
                    if hs == as_:
                        hs += 1
                    row.update(home_pace=rng.normal(99, 3), away_pace=rng.normal(99, 3),
                               home_ortg=112 + 3 * strength[h] + rng.normal(0, 4),
                               away_ortg=112 + 3 * strength[a] + rng.normal(0, 4))
                else:  # volleyball - 세트 스코어
                    ps = 1 / (1 + np.exp(-0.8 * d))
                    hs = as_ = 0
                    while hs < 3 and as_ < 3:
                        if rng.random() < ps:
                            hs += 1
                        else:
                            as_ += 1
                    row.update(home_attack_eff=0.5 + 0.03 * strength[h] + rng.normal(0, .03),
                               away_attack_eff=0.5 + 0.03 * strength[a] + rng.normal(0, .03))
                row.update(home_score=hs, away_score=as_)
                rows.append(row)
    df = pd.DataFrame(rows)
    scraper = ScraperFactory.create(sport)
    return scraper.standardize(df)


# ══════════════════════════════════════════════════════════════════════
# 진입 함수
# ══════════════════════════════════════════════════════════════════════
def collect(sport: str, demo: bool = False) -> pd.DataFrame:
    df = generate_synthetic(sport) if demo else ScraperFactory.create(sport).collect()
    if df.empty:
        log.warning("[%s] 저장할 데이터가 없습니다. 미구현 종목이면 --demo 로 파이프라인만 검증하세요.", sport)
        return df
    df = (df.drop_duplicates(subset=["date", "home", "away"])
            .sort_values("date").reset_index(drop=True))
    df.to_csv(config.matches_file(sport), index=False, encoding="utf-8-sig")
    log.info("[%s] 총 %d경기 저장 → %s%s", sport, len(df), config.matches_file(sport),
             "  (⚠ 가상 데이터)" if demo else "")
    return df


def load_matches(sport: str) -> pd.DataFrame:
    path = config.matches_file(sport)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} 없음 - 먼저 `python main.py scrape --sport {sport}` 실행")
    return pd.read_csv(path, parse_dates=["date"])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    collect(sys.argv[1] if len(sys.argv) > 1 else "soccer")
