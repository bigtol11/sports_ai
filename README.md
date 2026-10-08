# sports_ai — 초고차원 진화형 스포츠 예측 AI 시스템 (마스터 설계도)

> 국내 체육진흥투표권(베트맨) **프로토 승부식**과 **토토(승무패·승1패·승5패·스페셜·매치)** 대상
> 축구 · 야구 · 농구 · 배구 전 종목 통합 예측/베팅 의사결정 시스템.
>
> 이 문서는 **사람 + Codex + Claude 에이전트가 공통으로 참조하는 단일 진실 원천(Single Source of Truth)** 이다.
> **에이전트는 [§2 Codex 즉시 실행 지침](#2-codex-즉시-실행-지침-codex-quick-exec-playbook)부터 실행한다.**
> 코드와 이 문서가 어긋나면 이 문서를 먼저 갱신하고 코드를 맞춘다.

---

## 목차
0. [빠른 시작](#0-빠른-시작)
1. [시스템 개요 & 수익화 전략](#1-시스템-개요--수익화-전략)
2. [Codex 즉시 실행 지침 (Codex Quick-Exec Playbook)](#2-codex-즉시-실행-지침-codex-quick-exec-playbook)
3. [아키텍처 & 데이터 흐름](#3-아키텍처--데이터-흐름)
4. [데이터 계약](#4-데이터-계약)
5. [모듈 상세 설계](#5-모듈-상세-설계)
6. [초고차원 분석 기법 기술 요구사항](#6-초고차원-분석-기법-기술-요구사항)
7. [자가 진화형 강화학습 파이프라인](#7-자가-진화형-강화학습-파이프라인)
8. [종목별 피처 엔지니어링 리스트](#8-종목별-피처-엔지니어링-리스트)
9. [검증 · 리스크 관리 규칙](#9-검증--리스크-관리-규칙)
10. [AI 에이전트(Codex / Claude) 협업 규약](#10-ai-에이전트codex--claude-협업-규약)
11. [로드맵 & 작업 백로그](#11-로드맵--작업-백로그)
12. [책임 있는 이용 & 법적 고지](#12-책임-있는-이용--법적-고지)

---

## 0. 빠른 시작

```bash
pip install -r requirements.txt

# 1) 수집 (실데이터: 축구=football-data.co.uk, 야구=MLB Stats API)
python main.py scrape --sport soccer
python main.py scrape --sport baseball
python main.py scrape --sport basketball --demo   # 수집기 미구현 종목 → 가상 데이터로 파이프라인만 검증

# 2) 학습 (종목별 모델 → models/{sport}.joblib)
python main.py train --sport soccer               # --sport all 가능

# 3) 프로토 분석 (data/input/proto_odds.csv)
python main.py proto --bankroll 100000

# 4) 토토 마킹 (data/input/toto_{type}.csv)
python main.py toto --type soccer_wdl --budget 9600

# 5) 상태 점검 + 완료 메시지 (에이전트 작업 완료 확인용)
python main.py check
```

| 종목 | 리그 | 수집기 상태 |
|---|---|---|
| 축구 | EPL, 라리가, 분데스리가, 세리에A, 리그1 (+챔피언십/에레디비시/포르투갈) | ✅ 실동작 |
| 축구 | K리그1 | ⬜ 스켈레톤 |
| 야구 | MLB | ✅ 실동작 (결과 + 예고 선발 ID) |
| 야구 | KBO | ⬜ 스켈레톤 |
| 농구 | NBA, KBL | ⬜ 스켈레톤 (`--demo` 로 검증) |
| 배구 | V리그 남/여 | ⬜ 스켈레톤 (`--demo` 로 검증) |

> ⚠ `--demo` 데이터는 잠재 전력값으로 만든 **가상 경기**다. 실전 예측에 절대 사용하지 않는다.

---

## 1. 시스템 개요 & 수익화 전략

### 1.1 목표 정의 — "적중률"이 아니라 "기대가치"
- **적중률(accuracy)은 목표가 아니다.** 1.20 배당 정배만 고르면 적중률 80%여도 장기 손실이 난다.
- 시스템의 단일 목표 함수는 **장기 자금 성장률 = E[log(자금)]** 이며, 이는 다음 두 조건을 동시에 요구한다.
  1. **보정된 확률(calibration)**: 모델이 60%라고 하면 실제로 60% 일어나야 한다 → log loss / Brier 로 측정.
  2. **시장 대비 우위(edge)**: 모델 확률 × 배당 − 1 > 0 인 경우가 실제로 존재해야 한다.
- 따라서 모든 모델 평가의 **기준선(baseline)은 북메이커 내재 확률**이다. `train` 출력의 `bookmaker` 행보다
  log loss 가 낮지 않으면 그 모델은 "엣지 없음"으로 간주하고 실전 베팅 금액을 0으로 둔다.

> **현재 상태 (2026-10-09 기준 실측)**
> 축구 `clf+odds` log loss 0.9924 vs `bookmaker` 0.9854 → **아직 시장을 이기지 못함.**
> 프로토 추천 결과는 파이프라인 검증용이며, §9 의 승격 조건을 통과하기 전까지 실제 금액 베팅에 쓰지 않는다.

### 1.2 전 종목 통합 수익화 전략
| 상품 | 수익 구조 | 시스템 활용 방식 |
|---|---|---|
| **프로토 승부식** | 고정 배당 (마진 큼, 대략 10%+) | 모델 확률 vs 배당 → EV 필터 → 분수 켈리 베팅 |
| **토토 승무패/승1패/승5패** | 패리뮤추얼 (적중자끼리 분배) | ① 적중 확률 최대화 마킹 ② **대중 투표율 대비 과소평가 결과(역배 가치)** 를 섞어 기대 배당 증대 |
| **토토 스페셜/매치** | 패리뮤추얼, 점수 구간 | 몬테카를로 점수 분포 → 구간 확률 → 동일 DP 마킹 |

종목별 엣지 탐색 우선순위 (시장 효율성이 낮은 곳부터):
1. **국내 리그 (KBO·KBL·V리그·K리그)** — 해외 북메이커 관심이 적어 배당 비효율 가능성이 상대적으로 큼.
2. **언더/오버·핸디캡** — 승패보다 베트맨 배당 산정이 거친 경우가 있음 → 몬테카를로 점수 분포가 무기.
3. **유럽 5대리그 승무패** — 가장 효율적인 시장. 엣지 확보가 가장 어려움 (현재 실측도 이를 확인).

### 1.3 다폴더 억제 · 1~2폴더 집중 투자 원칙
**규칙 (코드로 강제됨: `config.PROTO_MAX_FOLDERS = 2`)**
- 조합은 **최대 2폴더**. 3폴더 이상은 생성 자체를 하지 않는다.
- 한 조합 안에 **같은 경기 2픽 금지** (상관관계로 확률 곱셈 가정이 깨짐).
- 같은 경기의 여러 가치 픽 중 **EV 최상위 1개만** 사용 (`best_per_match`).

**근거 (수학)**
- 프로토 마진을 폴더당 m 이라 하면, n폴더 조합의 사실상 마진은 `1 − (1/(1+m))^n` 으로 **복리 증가**한다.
  m=10% 기준: 1폴더 9.1% → 2폴더 17.4% → 3폴더 24.9% → 5폴더 37.9%.
- 모델 확률 오차 ε 도 곱셈으로 누적된다. 폴더 수가 늘면 EV 추정의 분산이 폭증해 "가짜 엣지"가 대부분을 차지하게 된다.
- 켈리 비율은 조합 확률이 낮을수록 급감 → 다폴더는 자금 성장률 기여가 거의 없고 파산 위험만 키운다.

**자금 관리 (기본값, `config.py`)**
| 항목 | 값 | 의미 |
|---|---|---|
| `KELLY_FRACTION` | 0.25 | 풀켈리의 1/4 (모델 오차 대비) |
| `MAX_STAKE_RATIO` | 5% | 1회 베팅 상한 |
| `MAX_DAILY_EXPOSURE` | 15% | 하루 총 베팅 상한 (초과 시 비례 축소) |
| `MIN_EV` | 3% | 이 미만은 추천하지 않음 |
| `MIN_PROB` | 5% | 극단 저확률 픽 제외 |

---

## 2. Codex 즉시 실행 지침 (Codex Quick-Exec Playbook)

> **Codex / Copilot / Claude 에이전트는 이 섹션부터 실행한다.** 사람의 추가 지시 없이 §2.1 루프를 돌며
> §2.2 작업 큐를 위에서부터 하나씩 완료한다. 각 작업의 "완료 조건"을 만족하지 못하면 다음 작업으로 넘어가지 않는다.

### 2.1 자율 작업 루프 (매 작업 공통)
```
1. 읽기     : README §2(이 섹션), §4(데이터 계약), §10(불변 규칙)
2. 선택     : §2.2 작업 큐에서 상태가 ⬜ 인 가장 위 작업 1개
3. 탐색     : grep -rn "TODO(<태그>)" *.py  → 수정 위치 확인
4. 소스 확인: §2.3 카탈로그의 엔드포인트를 실제로 1회 호출해 구조가 문서와 같은지 확인
              (다르면 구현 전에 이 README 의 해당 행을 먼저 고친다)
5. 구현     : §2.4 절차 + §2.5 레시피. 원본은 data/raw/{sport}/ 캐시
6. 검증     : python main.py scrape --sport <종목> → python main.py train --sport <종목> → python main.py check
7. 보고     : §2.7 형식으로 결과 보고, 작업 큐 상태를 ✅ 로 갱신
```

### 2.2 작업 큐 (위에서부터 순서대로)
| # | 상태 | 태그 | 작업 | 완료 조건 (모두 충족) |
|---|---|---|---|---|
| T1 | ⬜ | `TODO(선발투수)` | MLB 선발투수 **경기 전 누적** ERA/WHIP/K9 채우기 | `baseball_matches.csv` 의 `home_sp_era` 결측률 < 10% (시즌 첫 등판 제외) · `train --sport baseball` 무에러 · 지표표에 변경 전/후 log loss |
| T2 | ⬜ | `TODO(xG)` | Understat xG → 축구 `home_xg/away_xg` 병합 | 5대리그 2014+ 시즌 `home_xg` 결측률 < 5% · 팀명 매핑 실패 0건(로그) · soccer log loss 변경 전/후 기록 |
| T3 | ⬜ | `TODO(KBO)` | KBO 일정/결과 수집기 | 최근 3시즌 정규시즌 경기 수 = 720 ± 취소 경기 · 더블헤더 구분 · 우천취소 제외 |
| T4 | ⬜ | `TODO(배당)` | 해외 배당 수집 (The Odds API) | `ODDS_API_KEY` 환경변수 사용(코드에 키 금지) · 스냅샷 parquet 저장 · 무키 시 경고 후 건너뜀 |
| T5 | ⬜ | `TODO(NBA)` | NBA 결과 + PACE/ORtg/DRtg + 쿼터 득점 | 최근 3시즌 정규시즌 1,230경기/시즌 · `home_pace` 결측률 < 2% |
| T6 | ⬜ | `TODO(KBL)` | KBL 수집기 | 최근 3시즌 정규시즌 경기 전수 · 쿼터 득점 포함 |
| T7 | ⬜ | `TODO(KOVO)` | V리그 남/여 수집기 | 세트 스코어 + 세트별 점수 + 공격/리시브 효율 |
| T8 | ⬜ | `TODO(K리그)` | K리그1 수집기 | 최근 3시즌 228경기/시즌 |
| T9 | ⬜ | (신규) | `ledger.py` 베팅 기록·정산·CLV | §7.1 settle 단계 입력/출력 스키마 준수 |

### 2.3 데이터 소스 카탈로그 (2026-10-09 실제 호출로 확인)
| 소스 | 종목 | 접근 방식 | 엔드포인트 / URL | 확인 결과 | 비고 |
|---|---|---|---|---|---|
| football-data.co.uk | 축구 | Requests CSV | `/mmz4281/{2526}/{E0}.csv` | ✅ 구현됨 | 결과·슈팅·코너·배당(피나클 종가 포함) |
| **Understat** | 축구 xG | Requests JSON (XHR) | `GET https://understat.com/getLeagueData/{리그}/{시작연도}` | ✅ 200, JSON | 헤더 `X-Requested-With: XMLHttpRequest` + `Referer` 필수. 리그: `EPL, La_liga, Bundesliga, Serie_A, Ligue_1` |
| FBref | 축구 | – | `fbref.com/en/comps/...` | ⚠ **403 (봇 차단)** | 차단 우회(헤더 위장 반복·프록시 회전·캡차 해결) **금지**. xG 는 Understat 사용 |
| MLB Stats API | 야구 | Requests JSON | `statsapi.mlb.com/api/v1/schedule?sportId=1&season=…&hydrate=probablePitcher` | ✅ 구현됨 | 공개 API |
| MLB Stats API | 야구 선발 | Requests JSON | `/api/v1/people/{id}/stats?stats=gameLog&group=pitching&season=…` | ✅ 200 | 필드: `date, isHome, stat.{gamesStarted, inningsPitched, earnedRuns, hits, baseOnBalls, strikeOuts, homeRuns}` |
| **KBO 공식** | 야구 | Requests POST JSON (ASP.NET) | `POST https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList` | ✅ 200, JSON (헤더 필요) | 헤더 없으면 HTML 반환. §2.5-B |
| stats.nba.com | 농구 | `nba_api` 패키지 | `LeagueGameFinder`, `BoxScoreAdvancedV3`, `BoxScoreSummaryV2` | ⬜ 미확인 | `pip install nba_api` 후 확인. 요청 간격 ≥ 0.6초 |
| KBL 공식 | 농구 | DevTools 로 XHR 탐색 → 없으면 Selenium | `https://www.kbl.or.kr/` | ⬜ 200, 구조 미확인 | §2.4 절차대로 탐색 |
| KOVO 공식 | 배구 | 〃 | `https://www.kovo.co.kr/` | ⬜ 301 리다이렉트, 구조 미확인 | 〃 |
| K리그 데이터포털 | 축구 | 〃 | `https://data.kleague.com/` | ⬜ 미확인 | 〃 |
| The Odds API v4 | 전 종목 배당 | Requests JSON | `/v4/sports/{sport_key}/odds?regions=eu&markets=h2h,spreads,totals&oddsFormat=decimal&apiKey=…` | ✅ 응답 (키 없으면 `MISSING_KEY`) | 유료/무료 쿼터. 키는 `ODDS_API_KEY` 환경변수 |
| 시중 라이브스코어 앱/사이트 | – | **사용하지 않음** | – | – | 대부분 약관상 자동 수집 금지 + 봇 차단. 필요하면 라이선스 API(예: API-Football) 계약 |
| 베트맨 | 프로토/토토 배당 | **수동 CSV 입력** | – | – | 자동 크롤링은 약관 확인 후에만 |

### 2.4 종목별 웹 크롤러 구현 절차 (Requests → BeautifulSoup → Selenium)
`BaseScraper` 에 공통 헬퍼가 이미 있다. **우선순위를 절대 거꾸로 쓰지 않는다** (Selenium 은 느리고 깨지기 쉽다).

| 순위 | 상황 | 헬퍼 | 판별 방법 |
|---|---|---|---|
| 1 | 사이트가 내부적으로 JSON 을 호출 | `self.get_json(url, cache_name)` / `self.post_json(url, data, headers)` | 크롬 DevTools → Network → **Fetch/XHR** 필터 → 페이지 새로고침 → 응답이 JSON 인 요청 찾기 → *Copy as cURL* 로 헤더·파라미터 확인 |
| 2 | 데이터가 HTML 원문에 있음 (서버 렌더링) | `self.get_soup(url, cache_name)` / 표는 `pd.read_html` | 페이지 소스 보기(Ctrl+U)에서 원하는 숫자가 검색되면 서버 렌더링 |
| 3 | JS 실행 후에만 데이터가 생기고 XHR 도 못 찾음 | `driver = self.browser()` + `WebDriverWait` | 1·2 모두 실패했을 때만 |

**단계별 절차**
```
Step 0  robots.txt 와 이용약관 확인 → 자동 수집 금지면 중단하고 README 카탈로그에 "사용 불가" 기록
Step 1  대상 페이지 1개에서 순위 1→2→3 순으로 판별
Step 2  원본 1건을 data/raw/{sport}/ 에 저장 (cache_name 지정) → 이후 파서 개발은 캐시로만 (서버 부하 0)
Step 3  파서 작성: 원본 → 표준 스키마 컬럼 (README §4.1). 날짜는 tz 제거한 날짜, 점수는 int
Step 4  fetcher 등록: Scraper.fetchers 의 리그 코드에 연결 (NotImplementedError 제거)
Step 5  전체 시즌 수집: REQUEST_DELAY(1초) 준수, 진행 중 시즌만 refresh
Step 6  품질 점검: 시즌별 경기 수 / 중복 (date, home, away) / 점수 결측 / 팀명 수 (리그 팀 수와 일치?)
Step 7  검증 명령 (§2.1-6)
```

**템플릿 1 — Requests JSON (1순위)**
```python
def fetch_xxx(self, league: str, season: str) -> pd.DataFrame | None:
    data = self.get_json(URL.format(season=season), cache_name=f"{league}_{season}.json",
                         refresh=self.is_current(season), headers={"Referer": REFERER})
    if not data:
        return None
    rows = [{"date": g["date"], "home": g["home"], "away": g["away"],
             "home_score": g["hs"], "away_score": g["as"]} for g in data["games"] if g["final"]]
    return pd.DataFrame(rows)
```

**템플릿 2 — BeautifulSoup (2순위)**
```python
soup = self.get_soup(url, cache_name=f"{league}_{season}.html", refresh=self.is_current(season))
for tr in soup.select("table.schedule tbody tr"):          # 선택자는 반드시 실제 페이지에서 확인
    tds = [td.get_text(strip=True) for td in tr.select("td")]
    ...
# 표가 단순하면: pd.read_html(io.StringIO(str(soup.select_one("table"))))[0]
```

**템플릿 3 — Selenium (3순위)**
```python
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = self.browser(headless=True)
try:
    driver.get(url)
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    html = driver.page_source                     # 렌더링된 HTML → BeautifulSoup 으로 파싱
    (config.RAW_DIR / self.sport / f"{league}_{season}.html").write_text(html, encoding="utf-8")
finally:
    driver.quit()
```

**종목 공통 함정 체크리스트**
- 인코딩: 국내 구형 사이트는 `euc-kr` 일 수 있음 → `resp.encoding` 확인.
- `RowSpan`: 표에서 날짜 셀이 병합되면 직전 날짜를 이어받아야 한다 (KBO 일정이 그렇다).
- 시간대: 해외 API 는 UTC. 국내 경기는 KST 날짜 기준으로 맞춘다.
- 취소/연기 경기: 점수 없는 행은 제외 (우천취소, 코로나 등).
- 더블헤더(야구): 같은 날 같은 매치업 2경기 → 중복 제거 키에 `game_no` 추가 검토.
- 팀명: 소스마다 다르다 → 소스별 매핑 dict 를 만들고 매핑 실패를 로그로 남긴다.
- **봇 차단(403/캡차)을 만나면 우회하지 말고 중단**하고 카탈로그에 기록한다.

### 2.5 종목별 구현 레시피

**A. 축구 xG — Understat (T2)**
```python
UNDERSTAT_LEAGUE = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
year = "20" + season[:2]                     # "2526" → "2025"
data = self.get_json(f"https://understat.com/getLeagueData/{UNDERSTAT_LEAGUE[league]}/{year}",
                     cache_name=f"understat_{league}_{season}.json", refresh=self.is_current(season),
                     headers={"X-Requested-With": "XMLHttpRequest",
                              "Referer": f"https://understat.com/league/{UNDERSTAT_LEAGUE[league]}/{year}"})
xg = pd.DataFrame([{"date": pd.to_datetime(m["datetime"]).normalize(),
                    "home_us": m["h"]["title"], "away_us": m["a"]["title"],
                    "home_xg": float(m["xG"]["h"]), "away_xg": float(m["xG"]["a"])}
                   for m in data["dates"] if m["isResult"]])
# → UNDERSTAT_TEAM_MAP 으로 football-data 팀명 변환 후 (date, home, away) 로 left merge
#   (날짜가 하루 어긋나는 경우 대비: 같은 시즌 내 (home, away) 매치업은 유일하므로 (season, home, away) 키 권장)
```

**B. KBO 일정/결과 — 공식 웹서비스 (T3)**
```python
resp = self.post_json(
    "https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList",
    data={"leId": "1", "srIdList": "0,9,6", "seasonId": season, "gameMonth": f"{month:02d}", "teamId": ""},
    headers={"X-Requested-With": "XMLHttpRequest",
             "Referer": "https://www.koreabaseball.com/Schedule/Schedule.aspx"})
# 응답: {"rows": [{"row": [셀, 셀, ...]}]}, 셀 = {"Text": html, "Class": "day"|"time"|"play"|"relay"|None, "RowSpan": ...}
#  - Class="day"  : "07.01(화)"  (RowSpan 으로 이후 행에서는 생략됨 → 직전 값 유지)
#  - Class="play" : 원정팀명 <em><span class="win|lose">점수</span><span>vs</span><span class="...">점수</span></em> 홈팀명
#                   → BeautifulSoup(cell["Text"], "lxml") 로 팀명/점수 추출 (KBO 표기는 원정이 앞)
#  - Class="relay": 링크의 gameId (예: 20250701SSOB0 = 날짜 + 원정코드 + 홈코드 + 더블헤더번호)
#  - 마지막 셀 비고: 우천취소 등 → 제외
# 월 3~10 반복, seasonId=연도. srIdList 0=정규시즌 (9,6 은 포스트시즌 계열 - 학습 분리 권장)
```

**C. MLB 선발투수 경기 전 누적 스탯 (T1)**
```python
log = self.get_json(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group=pitching&season={season}",
                    cache_name=f"MLB_pitcher_{pid}_{season}.json", refresh=self.is_current(season))
splits = log["stats"][0]["splits"]           # 각 split: date, isHome, stat{inningsPitched, earnedRuns, hits, baseOnBalls, strikeOuts}
def ip_to_float(ip: str) -> float:           # "5.2" = 5와 2/3 이닝 (소수 아님!)
    whole, frac = ip.split(".")
    return int(whole) + int(frac) / 3
# 경기일 d 의 피처 = date < d 인 split 만 누적:
#   sp_era  = 9 * ΣER / ΣIP,  sp_whip = (ΣH + ΣBB) / ΣIP,  sp_k9 = 9 * ΣK / ΣIP
#   ΣIP < 10 이면 리그 평균으로 수축: (ΣER + 4.2*10/9) 형태의 베이지안 보정 권장
# 투수 수가 많으므로 pid 별 1회 호출 + 캐시 (시즌당 ~700명 × 1초)
```

**D. NBA — nba_api (T5)**
```python
from nba_api.stats.endpoints import leaguegamefinder   # pip install nba_api
games = leaguegamefinder.LeagueGameFinder(season_nullable="2024-25", league_id_nullable="00",
                                          season_type_nullable="Regular Season").get_data_frames()[0]
# 한 경기 = 팀별 2행 (MATCHUP 에 "vs." = 홈, "@" = 원정) → GAME_ID 로 피벗해 home/away 1행으로
# PACE/OFF_RATING/DEF_RATING: 경기별 고급 박스스코어 엔드포인트, 쿼터 득점: 박스스코어 요약의 LineScore
# 엔드포인트 이름·컬럼은 설치한 nba_api 버전에서 먼저 확인 (버전별로 V2/V3 가 다름)
```

**E. KBL / KOVO / K리그 (T6~T8)**: 카탈로그 상 구조 미확인 → §2.4 Step 0~1 로 1순위(XHR) 엔드포인트부터 찾고,
찾은 구조를 **이 README §2.3 표에 먼저 기록**한 뒤 구현한다.

**F. 해외 배당 — The Odds API (T4)**
```python
key = os.environ.get("ODDS_API_KEY")
if not key:
    log.warning("ODDS_API_KEY 미설정 - 배당 수집 건너뜀"); return pd.DataFrame()
url = (f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
       f"?regions=eu&markets=h2h,spreads,totals&oddsFormat=decimal&apiKey={key}")
# sport_key 예: soccer_epl, baseball_mlb, basketball_nba (GET /v4/sports 로 목록 확인)
# 응답 헤더 x-requests-remaining 로 쿼터 추적
```

### 2.6 Codex 프롬프트 예시 (복사해서 사용)

**P1 — 작업 큐 자동 진행**
```
README.md §2 를 읽고 §2.2 작업 큐에서 상태가 ⬜ 인 첫 작업을 수행해.
규칙: README §10.2 불변 규칙 준수, 상수는 config.py, 원본은 data/raw/{sport}/ 캐시.
구현 전 §2.3 카탈로그의 엔드포인트를 1회 실제 호출해 구조가 문서와 같은지 확인하고, 다르면 README 를 먼저 고쳐.
끝나면 python main.py scrape/train --sport <종목>, python main.py check 를 실행하고 §2.7 형식으로 보고해.
```

**P2 — 특정 수집기 구현 (예: KBO)**
```
scraper.py 의 BaseballScraper.fetch_kbo 를 구현해. 사양은 README §2.5-B.
- self.post_json 사용, 월(3~10) 단위 호출, 월별 원본을 data/raw/baseball/KBO_{season}_{month}.json 캐시
- play 셀 HTML 은 BeautifulSoup 으로 파싱 (원정팀이 앞), RowSpan 날짜 이어받기, 우천취소 제외
- 반환: date, home, away, home_score, away_score (표준 스키마 §4.1)
완료 조건: README §2.2 T3 행. 시즌별 경기 수를 로그로 출력해.
```

**P3 — 새 사이트 구조 탐색 (KBL/KOVO 등)**
```
kbl.or.kr 의 경기 일정/결과 페이지에서 데이터가 어디서 오는지 조사해.
순서: robots.txt 확인 → 페이지 HTML 에 점수가 있는지 → 없으면 JS 번들/네트워크에서 JSON API 탐색.
코드는 아직 쓰지 말고, 찾은 엔드포인트·파라미터·응답 예시를 README §2.3 표와 §2.5 에 레시피로 추가해.
봇 차단을 만나면 우회하지 말고 그 사실만 기록해.
```

**P4 — 피처 연결 후 성능 비교**
```
README §8 의 🔶 상태 피처 중 방금 데이터가 채워진 것을 확인하고, train 을 실행해 변경 전/후 지표표를 비교해.
북메이커 기준선 대비 개선이 없으면 그 사실을 그대로 보고하고 하이퍼파라미터를 임의로 늘리지 마.
```

### 2.7 보고 형식 (에이전트 → 사람)
```
[작업] T3 KBO 수집기
[변경 파일] scraper.py (+120 -3), README.md (§2.2 T3 ✅, §2.3 KBO 행 갱신)
[데이터] 2023: 720경기, 2024: 720경기, 2025: 716경기 (우천취소 4)
[검증] scrape ✅ / train ✅ / check ✅
[지표] baseball:clf log loss 0.6893 → 0.6871 (기준선 없음: KBO 배당 미수집)
[남은 이슈] 더블헤더 2차전 gameId 끝자리 2 처리 확인 필요
```

---

## 3. 아키텍처 & 데이터 흐름

```
sports_ai/
├── config.py            # 전역 설정: SPORTS 스펙, 하이퍼파라미터, 자금관리, TOTO_GAME_TYPES, TEAM_ALIASES
├── scraper.py           # [수집]  ScraperFactory → 종목별 Scraper → 표준 스키마
├── model.py             # [모델]  FeatureBuilder(종목별) → XGB+LGBM 앙상블 → MonteCarloSimulator
├── proto_analyser.py    # [프로토] 확률 부착 → EV/켈리 → 1~2폴더 포트폴리오
├── toto_scheduler.py    # [토토]  라인 확률 → DP 최적 복식 마킹 → 슬립지
├── main.py              # CLI 진입점
├── README.md            # ★ 이 문서 (마스터 설계도)
├── AGENTS.md / CLAUDE.md# 에이전트용 진입 안내 (→ README §10)
├── data/
│   ├── raw/{sport}/     # 원본 캐시 (CSV/JSON)
│   ├── processed/       # {sport}_matches.csv (표준 스키마)
│   └── input/           # 사용자 입력: proto_odds.csv, toto_{type}.csv
├── models/              # {sport}.joblib (SportModel 통째로 저장)
├── output/              # 분석 결과 CSV / 슬립지 TXT
└── logs/                # 베팅·예측 기록 (자가진화 파이프라인 입력, §7)
```

```
 [수집]                [피처/학습]                         [의사결정]
 ScraperFactory ──▶ {sport}_matches.csv ──▶ FeatureBuilder ──┬─▶ EnsembleClassifier ─▶ p(H/D/A) ─┬─▶ proto_analyser ─▶ 포트폴리오
  ├ Soccer (FD)                          (시간순 상태 누적)   │   (배당有 / 배당無 2종)            │
  ├ Baseball (MLB API)                                        └─▶ EnsembleRegressor ─▶ μ득점 ─▶ MonteCarlo ─┴─▶ toto_scheduler ─▶ 슬립지
  ├ Basketball (TODO)                                                                   (1만 회)
  └ Volleyball (TODO)
 OddsScraper / SentimentCollector (TODO) ──▶ 얼터너티브 피처 (§6-③)
 ST-GNN (TODO) ──▶ 팀/선수 임베딩 피처 (§6-②)
 Nightly Evolver (TODO) ──▶ 재학습 · 챔피언/챌린저 · DQN 스테이킹 (§7)
```

**핵심 설계 원칙**
1. **역할 분리**: 수집 / 모델 / 프로토 / 토토 모듈은 서로의 내부를 모른다. 연결은 DataFrame 데이터 계약(§4)으로만 한다.
2. **데이터 누수 0**: 모든 피처는 "경기 시작 시점에 알 수 있었던 정보"만 사용. `FeatureBuilder` 가 시간순으로 한 경기씩
   `features_for()`(읽기) → `update()`(쓰기) 순서를 강제한다. 무작위 train/test 분할 금지 (항상 시간순).
3. **종목 확장 = 설정 + 등록**: 새 종목/리그는 `config.SPORTS` 항목 + Scraper fetcher + (선택) FeatureBuilder 하위 클래스만 추가.
4. **하나의 점수 분포가 모든 마켓을 만든다**: 승패·핸디캡·언오버·승1패·승5패·스페셜·세트스코어 모두
   몬테카를로 점수 샘플(`SimResult`) 하나에서 파생 → 마켓 간 확률이 서로 모순되지 않는다.

---

## 4. 데이터 계약

### 4.1 표준 경기 스키마 — `data/processed/{sport}_matches.csv`
| 컬럼 | 타입 | 설명 |
|---|---|---|
| `date` | datetime | 경기일 (UTC 기준 날짜) |
| `sport` | str | soccer / baseball / basketball / volleyball |
| `league` | str | 리그 코드 (`config.SPORTS[sport]["leagues"]` 키) |
| `season` | str | 시즌 코드 |
| `home`, `away` | str | 팀명 (데이터 소스 영문 표기, 한글은 `TEAM_ALIASES` 로 매핑) |
| `home_score`, `away_score` | int | 최종 점수 (배구는 **세트 스코어**) |
| `odds_home`, `odds_draw`, `odds_away` | float | 해외 기준 배당 (없으면 NaN) |
| `home_<stat>`, `away_<stat>` | float | `rolling_stats` = **경기 후** 기록 (팀별 최근 N경기 평균으로만 사용) |
| `home_<p>`, `away_<p>` | float | `prematch` = **경기 전** 확정값 (선발 ERA, 부상 비중 등, 그대로 피처) |

> ⚠ `rolling_stats` 와 `prematch` 를 혼동하면 누수가 발생한다. 경기 결과 이후에야 알 수 있는 값을 `prematch` 에 넣지 말 것.

### 4.2 프로토 입력 — `data/input/proto_odds.csv` (long 포맷, 1행 = 1선택지)
```
date,sport,league,home,away,market,line,selection,odds
2026-10-17,soccer,E0,Arsenal,Chelsea,WDL,,H,1.85
2026-10-17,soccer,E0,Arsenal,Chelsea,OU,2.5,O,1.90
2026-10-17,soccer,E0,Liverpool,Tottenham,HANDICAP,-1.0,H,2.60
```
| market | selection | line | 확률 출처 |
|---|---|---|---|
| `WDL` | H / D / A | – | 분류기 |
| `ML` | H / A | – | 분류기 |
| `HANDICAP` | H / D / A | 홈 핸디캡 (예 −1.5, +1.0) | 몬테카를로 |
| `OU` | O / U | 기준 점수 | 몬테카를로 |

### 4.3 토토 입력 — `data/input/toto_{type}.csv`
```
game,date,league,home,away,pub_1,pub_2,pub_3
1,2026-10-17,E0,Arsenal,Chelsea,55,25,20
```
- `pub_i` (선택): i번째 선택지 대중 투표율(%). 있으면 역배 가치(`value_ratio`)·이변 경고(`upset_alert`) 산출.
- `type` 은 `config.TOTO_GAME_TYPES` 키: `soccer_wdl`, `baseball_w1l`, `basketball_w5l`, `soccer_special`,
  `baseball_special`, `basketball_special`, `volleyball_match`.
- ⚠ 승1패/승5패 기준 점수차, 스페셜 득점 구간, 회차당 경기 수는 **회차 공지마다 확인**하고 config 를 맞출 것.

### 4.4 출력 — `output/`
| 파일 | 내용 |
|---|---|
| `proto_singles.csv` | 전 선택지 prob / fair_odds / EV / kelly / stake / is_value |
| `proto_combos.csv` | 2폴더 조합 상위 (켈리 순) |
| `proto_portfolio.csv` | **최종 추천** (단폴 + 2폴더, 일일 한도 적용) |
| `toto_{type}_marking.csv` | 라인별 확률 / 마킹 / 단일·복수·올커버 / 역배 가치 |
| `toto_{type}_tickets.csv` | 실제 구매할 전체 조합 |
| `toto_{type}_slip.txt` | 슬립지 레이아웃 (■ 마킹 / □ 미마킹) + 분배율 + 비용 + 적중확률 |

---

## 5. 모듈 상세 설계

### 5.1 `scraper.py` — 팩토리 패턴 수집기
```python
class BaseScraper(ABC):
    sport: str
    fetchers -> dict[league_code, Callable[[league, season], DataFrame]]   # 하위 클래스가 정의
    http_get(url, cache_name, refresh)    # 캐시 + 지연 + 예외 처리 공통
    standardize(df)                        # 표준 스키마 강제
    collect()                              # 리그 × 시즌 순회, NotImplementedError 는 경고 후 건너뜀

ScraperFactory.create("baseball")  → BaseballScraper
ScraperFactory.register("esports", EsportsScraper)   # 확장
```
| 클래스 | fetcher | 소스 | TODO |
|---|---|---|---|
| `SoccerScraper` | `fetch_football_data` | football-data.co.uk CSV (결과, 슈팅, 유효슈팅, 코너, 피나클/평균 배당) | xG 병합 (understat/fbref) |
| | `fetch_kleague` | data.kleague.com | 전체 구현 |
| `BaseballScraper` | `fetch_mlb` | statsapi.mlb.com schedule (+probablePitcher) | 선발 **경기 전 누적** ERA/WHIP/FIP (gameLog), 배당 |
| | `fetch_kbo` | koreabaseball.com | 전체 구현 |
| `BasketballScraper` | `fetch_nba` / `fetch_kbl` | nba_api (LeagueGameFinder, BoxScoreAdvanced, LineScore) / kbl.or.kr | 전체 구현 |
| `VolleyballScraper` | `fetch_vleague` | kovo.co.kr | 전체 구현 |
| `OddsScraper` | `fetch_snapshots` | 해외 배당 API | §6-③ |
| `SentimentCollector` | `collect` | 뉴스/커뮤니티 | §6-③ |

**수집기 구현 규칙**: 요청 간 `REQUEST_DELAY` 준수, 원본은 `data/raw/{sport}/` 캐시, 진행 중 시즌만 재요청,
robots.txt·이용약관 확인. 베트맨 배당은 **사용자 수동 입력 CSV** 가 기본 (자동 크롤링은 약관 확인 후에만).

### 5.2 `model.py` — 종목별 ML 파이프라인
| 구성 | 설명 |
|---|---|
| `FeatureBuilder` | Elo(점수차 가중), 휴식일/백투백, 최근 N경기 폼(전체/홈/원정), rolling_stats 평균, prematch 값, 배당 내재확률 |
| `SoccerFeatureBuilder` | **무승부 변수**: `draw_tendency`(양팀 최근 무승부율), `low_scoring`(최근 총득점), `parity`(전력 균형) |
| `BaseballFeatureBuilder` | **선발투수 매치업**: `sp_matchup_home = 홈 타선 득점력 × 원정 선발 ERA/4` 등 |
| `BasketballFeatureBuilder` | **PACE×레이팅 기대득점**, `net_rtg_diff`, **쿼터별**(`clutch_diff` = 4쿼터 득점차) |
| `VolleyballFeatureBuilder` | `atk_vs_rcv` (공격 효율 vs 상대 리시브 효율 교차) |
| `EnsembleClassifier` | XGB + LGBM 확률 가중평균. 축구 3클래스(H/D/A), 그 외 2클래스(H/A) |
| `EnsembleRegressor` | 홈/원정 기대 득점 μ. 축구·야구는 포아송 목적함수 |
| `MonteCarloSimulator` | §6-① |
| `SportModel` | train / predict / simulate / save / load. **배당有 분류기 + 배당無 분류기** 2종을 자동 선택 |

학습 절차 (`SportModel.train`):
1. `build_history` → 시간순 피처, 이력 부족 초기 경기 제거
2. 마지막 `TEST_RATIO`(15%) 를 시간순 홀드아웃으로 평가 → **북메이커 기준선과 비교**
3. 평가 후 전체 데이터로 재학습해 저장

### 5.3 `proto_analyser.py`
`attach_probabilities` (종목별 모델 로드 → WDL/ML=분류기, HANDICAP/OU=몬테카를로) →
`analyse_singles` (EV, Kelly, stake) → `best_per_match` → `analyse_combos` (2폴더) → `build_portfolio` (일일 한도).

### 5.4 `toto_scheduler.py`
`build_lines` (게임 유형별 라인 확률) → `optimise_marking` (DP) → `build_report` → `expand_tickets` → `render_slip`.

**DP 정식화**: 라인 i 에서 상위 k_i 개 마킹, 비용 C = Πk_i ≤ 예산/단가, 목표 max Σ log(top-k_i 확률합).
상태 = 누적 조합 수(정수), 가능한 상태 수가 작아 **전역 최적해**를 즉시 구한다.
"복식 마킹 분배율" = 단일/복수/올커버 개수 (예: `단일 8 / 복수 5 / 올커버 1 → 2^5 × 3 = 96조합`).

**확장 TODO**: 패리뮤추얼 기대 배당 최적화 — 목표를 `P(적중) × E[배당 | 적중]` 로 바꾸고,
`E[배당]` 을 대중 투표율로 추정해 "모두가 고르는 정배"를 일부 회피 (역배 가치 반영).

---

## 6. 초고차원 분석 기법 기술 요구사항

### ① 상황별 몬테카를로 시뮬레이션 엔진 (1만 회 가상 경기)

**현재 구현 (v0, `model.MonteCarloSimulator`)**
| 종목 | 점수 생성 모델 | 동점 처리 |
|---|---|---|
| 축구 | 독립 포아송(μ_home, μ_away) | 무승부 허용 |
| 야구 | 음이항 NB(μ, r=`nb_dispersion`) — 과산포 반영 | 이닝당 μ/9 포아송 연장, 리그별 `max_extra_innings` (KBO 무승부 가능) |
| 농구 | 정규분포(μ, σ=`score_std`) | 5분 연장(μ×5/48) 반복 |
| 배구 | 매치 승률 → **세트 승률 역산**(이분법) → 5세트 3선승 시뮬 | 없음 |

`SimResult` 가 제공하는 마켓: `wdl()`, `handicap(line)`, `over_under(line)`, `margin(m)`(승1패/승5패),
`score_bucket(side, buckets)`(스페셜), `set_score(labels)`(배구).
수렴 오차: n=10,000 에서 확률 p 의 표준오차 = √(p(1−p)/n) ≤ 0.5%p.

**목표 구조 (v1 — "상황별")**
```
ScenarioSpec ──▶ ParameterAdjuster ──▶ GenerativeEngine(sport) ──▶ SimResult ──▶ 마켓 확률
 (상황 정의)       (μ, σ, 세트확률 보정)    (종목별 생성 모델)          (점수 샘플)
```
- `ScenarioSpec` 예시: `{"home_missing": ["주전 GK"], "weather": "rain", "sp_change": True, "rest_home": 2}`
- `ParameterAdjuster`: 시나리오별 보정 계수는 **과거 데이터에서 추정**(하드코딩 금지). 예) 주전 결장 시 μ 변화 =
  해당 선수 출전/결장 경기의 득실 차이를 계층 베이지안으로 축소 추정.
- 시나리오 가중 혼합: `P(마켓) = Σ_s P(s) × P(마켓 | s)` — 예) 선발 미확정이면 후보 투수별 확률 가중.
- 종목별 고해상도 생성 모델 (v2):
  | 종목 | 모델 | 상태 공간 |
  |---|---|---|
  | 축구 | Dixon-Coles / 이변량 포아송 → 분 단위 강도함수(시간대별 득점률, 레드카드 이벤트) | (분, 스코어, 인원) |
  | 야구 | **타석 단위 마르코프 체인** (24 base-out 상태), 투수 교체 = 투구수/피로 규칙 | (이닝, 아웃, 주자, 투수) |
  | 농구 | **포제션 단위 시뮬** (PACE 로 포제션 수, ORtg/DRtg 로 득점 확률, 쿼터별 보정) | (쿼터, 잔여시간, 점수차) |
  | 배구 | **랠리 포인트 모델** (서브권 측 득점 확률 p_serve / 리시브 측 p_receive, 25점·듀스·5세트 15점) | (세트, 점수, 서브권) |
- 성능 요구: 경기당 10,000회 < 0.5초 (NumPy 벡터화, 필요 시 Numba). 시드 고정으로 재현성 보장.
- 검증: 시뮬 분포의 득점 평균/분산/무승부율/언오버 적중률을 과거 실측과 **캘리브레이션 플롯**으로 대조.

### ② 시공간 그래프 신경망 (ST-GNN) — 종목 간 · 선수 간 상성 분석

**그래프 정의 (경기일 t 마다 스냅샷 G_t = (V, E_t, X_t))**
| 노드 타입 | 노드 피처 X (예시) |
|---|---|
| `player` | 포지션, 최근 N경기 스탯(종목별 §8), 출전 시간, 부상 상태, 나이, 피로(최근 7일 출전량) |
| `team` | Elo, 폼, 홈/원정, 휴식일, 감독 ID 임베딩 |
| (선택) `venue` | 파크팩터(야구), 고도, 잔디 종류 |
| (선택) `referee` | 카드/파울 성향 |

| 엣지 타입 | 연결 | 가중치/속성 |
|---|---|---|
| `teammate` | 같은 팀, 함께 출전 | 동시 출전 시간 비율 |
| `matchup` | 직접 맞대결 관계 | 야구: 투수↔타자 상대 전적(피OPS) / 농구: 매치업 수비 / 축구: 포지션 대응(윙어↔풀백) / 배구: 서버↔리시버 |
| `plays_for` | player → team | 출전 비중 |
| `opponent` | team ↔ team | 해당 경기 홈/원정 방향 |
| `temporal` | node(t−1) → node(t) | 시간 간격(일) |

**모델 구조**
```
X_t ──▶ [공간] 이종 그래프 어텐션(HeteroGAT/HGT, 2~3층) ──▶ 노드 임베딩 h_t
h_{t-k..t} ──▶ [시간] GRU 또는 Temporal Conv ──▶ 시간 인식 팀 임베딩 z_team
(z_home, z_away, z_home ⊙ z_away) ──▶ MLP 헤드 ──▶ p(H/D/A), μ_home, μ_away
```
- 라이브러리: PyTorch Geometric (`HeteroData`, `HGTConv`) + PyTorch Geometric Temporal.
- **연동 방식 2가지**: (a) 임베딩 z 를 XGB/LGBM 피처로 추가(권장 시작점), (b) 독립 모델로 앙상블에 합류.
- **종목 간 상성**: 종목별 그래프는 분리 학습하되, 공통 메타 피처(휴식, 이동거리, 홈 이점)를 공유하는
  멀티태스크 헤드로 저데이터 종목(배구) 의 학습을 보조.
- 누수 규칙: 스냅샷 G_t 는 t 이전 경기로만 구성. 라인업은 **공식 발표 시점 이후**에만 노드 활성화.
- 콜드스타트: 신규 선수 노드는 포지션 평균 임베딩으로 초기화.

### ③ 얼터너티브 데이터 연동 — 해외 배당 흐름 + 미디어/SNS 감성 분석

**(a) 배당 흐름 (Odds Movement)**
- 저장 스키마 `data/raw/odds/{sport}_{date}.parquet`: `match_key, ts, book, market, selection, odds`
- 수집 주기: 경기 72h 전부터 6h 간격 → 6h 전부터 30분 간격 → 킥오프 직전 종가.
- 피처:
  | 피처 | 정의 |
  |---|---|
  | `drift_open_close` | (종가 내재확률 − 오픈 내재확률) |
  | `steam_flag` | 짧은 시간(≤15분) 다수 북메이커 동시 2%p+ 이동 |
  | `book_dispersion` | 북메이커 간 내재확률 표준편차 (불확실성) |
  | `sharp_soft_gap` | 피나클(샤프) vs 소프트북/베트맨 내재확률 차 → **베트맨 오배당 탐지의 핵심** |
- 평가 지표 **CLV(Closing Line Value)** = 베팅 배당 / 종가 배당 − 1. 장기 CLV > 0 은 엣지의 가장 신뢰성 높은 선행 지표.

**(b) 감성 분석 (Sentiment)**
- 소스: 스포츠 뉴스 RSS, 구단 공식 SNS, 커뮤니티 게시판 (약관 허용 범위).
- 파이프라인: 수집 → 팀/선수 개체 인식(별칭 사전) → 감성·이벤트 분류 → 시간 창 집계.
- 모델: 한국어 `klue/roberta-base` 파인튜닝 또는 LLM 분류 (라벨: 긍정/부정/중립 + 이벤트: 부상·불화·감독교체·로테이션 예고).
- 피처: `sent_mean_24h`, `sent_volume_z`(평소 대비 언급량 z-score), `injury_news_flag`, `rotation_hint_flag`.
- **시간 절단 규칙**: 피처 계산은 반드시 `kickoff − 60분` 이전 텍스트만 사용 (결과 기사 누수 방지).

---

## 7. 자가 진화형 강화학습 파이프라인

### 7.1 Nightly 온라인 러닝 (매일 밤 자동)
```
02:00  collect      : 전날 경기 결과·스탯·종가 배당 수집 → {sport}_matches.csv append
02:20  settle       : logs/bets.csv 의 베팅 정산 (적중/미적중, 손익, CLV)
02:30  evaluate     : 최근 30/90일 rolling log loss, Brier, 캘리브레이션, CLV, ROI → logs/metrics.csv
02:40  fine-tune    : 챌린저 모델 생성
                      - LGBM: init_model=기존 부스터 + 신규 데이터로 트리 추가 (learning_rate 축소)
                      - XGB : xgb_model=기존 부스터 이어서 학습
                      - 주 1회는 전체 재학습(드리프트 누적 방지)
03:10  gate         : 챔피언 vs 챌린저 — 최근 홀드아웃에서 log loss 개선 & 기준선 대비 악화 없음 → 승격
03:20  rl-update    : DQN 에이전트 리플레이 버퍼에 오늘 전이 추가 → 소량 학습 스텝
03:30  report       : 요약 리포트 생성
```
- 스케줄러: Windows 작업 스케줄러 (`python evolve.py nightly`) — `evolve.py` 는 TODO.
- 모든 모델은 `models/{sport}/{YYYYMMDD}.joblib` 버전 관리, `models/{sport}.joblib` 은 챔피언 심볼.
- 롤백: 승격 후 7일 rolling log loss 가 이전 챔피언보다 나쁘면 자동 롤백.

### 7.2 DQN 베팅 에이전트 — "얼마나 걸 것인가" 를 학습
**역할 한정**: 확률 예측은 지도학습 모델이 담당하고, DQN 은 **스테이킹 정책(패스/비중)만** 학습한다.
리스크 한도(§1.3)는 에이전트가 넘을 수 없는 하드 제약(action masking)이다.

| 요소 | 정의 |
|---|---|
| **State** s | 모델 확률 p, 배당 o, EV, 켈리 f*, 모델 불확실성(앙상블 분산), `sharp_soft_gap`, 배당 흐름, 종목·마켓 원핫, 현재 자금/최고점 비율(드로다운), 오늘 누적 노출, 최근 30일 모델 캘리브레이션 오차 |
| **Action** a | 이산 5개: {패스, 0.25·f_cap, 0.5·f_cap, 0.75·f_cap, 1.0·f_cap} (f_cap = 분수켈리 상한). 2폴더 후보는 별도 에피소드 스텝 |
| **Action mask** | 일일 노출 초과, 3폴더 이상, 동일 경기 중복, EV < 0 이면 패스만 허용 |
| **Reward** r | `r = log(B_{t+1} / B_t)` (켈리와 일치하는 로그 자금 성장) |
| | `− λ_dd · max(0, DD_t − DD_limit)` (드로다운 15% 초과 페널티, λ_dd=2.0) |
| | `+ λ_clv · CLV · stake_ratio` (소량 보조 보상, λ_clv=0.1 — 결과 운에 덜 의존하는 신호) |
| 에피소드 | 하루 = 1 에피소드 (후보 픽을 EV 순으로 순차 결정) |
| 학습 | Double DQN + Dueling, 경험 리플레이, 타깃 네트워크(1,000 스텝 동기화), ε-greedy (1.0→0.05) |
| 오프라인 사전학습 | 과거 시즌 + 과거 배당으로 워크포워드 리플레이 (시즌 단위로 학습/검증 분리) |

**안전 장치 (필수)**
- 실자금 투입 전 **페이퍼 트레이딩 ≥ 500 베팅** 에서 고정 1/4 켈리 대비 우위 확인.
- 에이전트 정책이 고정 켈리보다 나쁘면 자동으로 고정 켈리로 폴백.
- 킬 스위치: 누적 드로다운 25% 도달 시 전 베팅 중단 + 사람 검토.

---

## 8. 종목별 피처 엔지니어링 리스트

상태 표기: ✅ 구현 / 🔶 스키마 준비(데이터 연결 필요) / ⬜ TODO

### 8.1 축구 (EPL · 5대리그 · K리그)
| 피처 | 정의 | 상태 |
|---|---|---|
| Elo (점수차 가중) | `elo_home/away/diff`, `elo_exp_home` | ✅ |
| 최근 폼 | 최근 6경기 승점/득점/실점 (전체·홈·원정) | ✅ |
| 휴식일수 | 직전 경기 후 경과일, 백투백 | ✅ |
| 슈팅 / 유효슈팅 / 코너 | 최근 6경기 평균 (`shots`, `sot`, `corners`) | ✅ |
| **무승부 변수** | `draw_tendency`, `low_scoring`, `parity` | ✅ |
| 배당 내재확률 | 마진 제거 `imp_H/D/A` | ✅ |
| **xG / xGA** | 최근 N경기 기대득점·기대실점 (`xg`) | 🔶 |
| **부상자 비중** | 결장 선수의 (출전시간 × 영향도) 합 / 팀 총합 (`injury_ratio`) | 🔶 |
| 라인업 강도 | 선발 11인 평균 레이팅 vs 시즌 베스트11 | ⬜ |
| 동기 지수 | 우승·강등·유럽대항전 경쟁 승점차, 시즌 잔여 경기 | ⬜ |
| 일정 밀도 | 최근 14일 경기 수, 주중 유럽대항전 여부 | ⬜ |
| 이동 거리 / 더비 | 원정 이동 km, 라이벌전 플래그 | ⬜ |
| 심판 성향 | 경기당 카드/PK | ⬜ |

### 8.2 야구 (KBO · MLB)
| 피처 | 정의 | 상태 |
|---|---|---|
| Elo / 폼 / 휴식 | 공통 | ✅ |
| **선발 ERA** | 해당 경기 **이전까지** 시즌 ERA (`sp_era`) | 🔶 |
| **선발 WHIP** | (피안타+볼넷)/이닝 (`sp_whip`) | 🔶 |
| 선발 FIP / K9 / 좌우 | `sp_fip`, `sp_k9`, `sp_hand` | 🔶 |
| **선발 매치업** | `sp_matchup_home/away` = 상대 타선 득점력 × 선발 ERA | ✅(ERA 연결 시 활성) |
| 불펜 ERA / 피로도 | `bullpen_era`, 최근 3일 불펜 투구 이닝 (`bullpen_ip`) | 🔶 |
| 타선 wOBA/OPS vs 좌·우투 | 상대 선발 투구 손 기준 | ⬜ |
| 파크 팩터 | 구장별 득점 보정 | ⬜ |
| 날씨 / 바람 | 기온, 풍향·풍속 (돔 여부) | ⬜ |
| 투수-타자 상대 전적 | ST-GNN `matchup` 엣지 (§6-②) | ⬜ |

### 8.3 농구 (NBA · KBL)
| 피처 | 정의 | 상태 |
|---|---|---|
| Elo / 폼 / 휴식 / **백투백** | 공통, `b2b_home/away` | ✅ |
| **PACE** | 48분당 포제션 (`pace`) | 🔶 |
| **ORtg / DRtg** | 100포제션당 득점/실점 (`ortg`, `drtg`) | 🔶 |
| 기대 득점 | `exp_pts = PACE × (ORtg + 상대 DRtg)/2 /100` | ✅(데이터 연결 시 활성) |
| Net Rating 차 | `net_rtg_diff` | ✅(〃) |
| **쿼터별 득점** | `q1~q4` 최근 평균, `clutch_diff`(4쿼터) | 🔶 |
| 3점 시도 비율 | `fg3_rate` (분산 ↑ → 이변 ↑) | 🔶 |
| 부상 (출전시간 가중) | `injury_ratio` | 🔶 |
| 원정 연속 경기 / 시차 | 연속 원정 수, 시간대 이동 | ⬜ |
| 탱킹·플레이오프 동기 | 순위 경쟁 지표 | ⬜ |

### 8.4 배구 (V리그)
| 피처 | 정의 | 상태 |
|---|---|---|
| Elo / 폼 / 휴식 | 공통 (세트 득실 기반) | ✅ |
| 공격 성공률 | `attack_eff` | 🔶 |
| 리시브 효율 | `receive_eff` | 🔶 |
| 블로킹 / 서브 에이스 | 세트당 `blocks`, `aces` | 🔶 |
| 공격-리시브 교차 | `atk_vs_rcv` | ✅(데이터 연결 시 활성) |
| 총 득점 | `points_total` (언오버용) | 🔶 |
| 외국인·아시아쿼터 결장 | 결장 시 공격 점유율 손실 | ⬜ |
| 세트 승률 → 매치 승률 | 몬테카를로 역산 (§6-①) | ✅ |

---

## 9. 검증 · 리스크 관리 규칙

1. **시간순 분할만 허용**. 최종 평가는 워크포워드(시즌 단위 롤링).
2. **기준선 비교 필수**: 모든 리포트에 `bookmaker` 행 포함. 기준선 미달 모델은 `stake=0`.
3. **실전 승격 조건 (모두 충족)**
   - 최근 2시즌 워크포워드 log loss < 북메이커 기준선
   - 페이퍼 트레이딩 ≥ 500 베팅, 평균 CLV > 0, ROI 95% 신뢰구간 하한 > −2%
   - 캘리브레이션 곡선 기울기 0.9~1.1
4. **확률 보정**: 필요 시 Isotonic/Platt (시간순 검증셋으로만 적합).
5. **기록**: 모든 추천·베팅은 `logs/` 에 (시각, 모델 버전, 확률, 배당, 금액, 결과) 기록 — 자가진화 입력.

---

## 10. AI 에이전트(Codex / Claude) 협업 규약

### 10.1 작업 전 필독
- **§2 Codex 즉시 실행 지침** (작업 루프 · 작업 큐 · 소스 카탈로그 · 레시피 · 보고 형식)
- 이 README 전체, 특히 §3(원칙), §4(데이터 계약), §9(검증).
- `config.py` — 모든 상수는 여기서만 정의.

### 10.2 불변 규칙 (위반 시 PR 반려)
1. **모듈 경계**: scraper 는 표준 스키마만 내보낸다 / model 은 확률·μ 만 내보낸다 / proto·toto 는 모델 내부를 참조하지 않는다.
2. **데이터 누수 금지**: 새 피처는 반드시 `FeatureBuilder.features_for()`(읽기)와 `update()`(쓰기)에 나눠 구현.
3. **하드코딩 금지**: 리그 코드, 임계값, 버킷 구간은 `config.py`.
4. **리스크 한도 우회 금지**: `PROTO_MAX_FOLDERS`, `MAX_DAILY_EXPOSURE`, `KELLY_FRACTION` 상향은 사람 승인 필요.
5. **스키마 변경 시** §4 문서 동시 갱신.
6. 미구현 부분은 `NotImplementedError` + `# TODO(<태그>):` 주석으로 남긴다 (조용히 빈 값 반환 금지).

### 10.3 검증 명령 (작업 완료 정의)
```bash
python main.py scrape --sport soccer && python main.py train --sport soccer   # 실데이터 회귀
python main.py scrape --sport basketball --demo && python main.py train --sport basketball
python main.py proto
python main.py toto --type soccer_wdl
python main.py check
```
- 5개 명령 모두 에러 없이 종료 + `train` 지표표에 `bookmaker` 행 출력(배당 있는 종목) = 완료.
- 성능 관련 변경은 변경 전/후 지표표를 PR 설명에 첨부.

### 10.4 새 종목/리그 추가 체크리스트
1. `config.SPORTS[...]` 항목 (outcomes, score_dist, rolling_stats, prematch, Elo 파라미터)
2. `scraper.py`: `BaseScraper` 하위 클래스 + `fetchers` + `ScraperFactory._registry` 등록
3. (선택) `model.py`: `FeatureBuilder` 하위 클래스 + `BUILDERS` 등록
4. (선택) `MonteCarloSimulator.run` 에 점수 분포 분기
5. (선택) `config.TOTO_GAME_TYPES` 에 토토 유형
6. `--demo` 로 파이프라인 검증 → 실데이터 수집 → §9 기준 평가

### 10.5 작업 태그
`TODO(xG)`, `TODO(선발투수)`, `TODO(배당)`, `TODO(KBO)`, `TODO(NBA)`, `TODO(KBL)`, `TODO(KOVO)`, `TODO(K리그)` — `grep -rn "TODO(" *.py` 로 전체 목록 확인.

---

## 11. 로드맵 & 작업 백로그

| 우선순위 | 작업 | 기대 효과 | 담당 모듈 |
|---|---|---|---|
| P0 | MLB 선발투수 경기 전 누적 스탯 연결 | 야구 모델 핵심 변수 활성화 | scraper |
| P0 | KBO 수집기 | 국내 리그(비효율 시장) 진입 | scraper |
| P0 | `logs/` 베팅 기록 + 정산 + CLV 계산 | 엣지 검증의 기반 | 신규 `ledger.py` |
| P1 | 축구 xG 병합, Dixon-Coles 무승부 보정 | 축구 log loss 개선 | scraper / model |
| P1 | NBA·KBL·V리그 수집기 | 전 종목 실데이터화 | scraper |
| P1 | 해외 배당 스냅샷 + `sharp_soft_gap` | 베트맨 오배당 탐지 | scraper / model |
| P1 | 토토 패리뮤추얼 기대 배당 최적화 | 토토 기대 수익 증대 | toto_scheduler |
| P2 | Nightly evolve (`evolve.py`) + 챔피언/챌린저 | 자가 진화 | 신규 |
| P2 | 상황별 몬테카를로 v1 (ScenarioSpec) | 라인업·날씨 반영 | model |
| P3 | ST-GNN 임베딩 피처 | 선수 상성 | 신규 `gnn/` |
| P3 | 감성 분석 피처 | 부상·불화 조기 탐지 | 신규 `altdata/` |
| P3 | DQN 스테이킹 에이전트 (페이퍼 트레이딩 우선) | 동적 자금 관리 | 신규 `rl/` |

---

## 12. 책임 있는 이용 & 법적 고지
- 국내에서 합법적인 스포츠 베팅은 **베트맨(체육진흥투표권)** 뿐이다. 불법 사설 사이트에 이 시스템을 사용하지 않는다.
- 만 19세 이상만 구매 가능. 구매 한도 등 베트맨 규정을 준수한다.
- 어떤 모델도 수익을 보장하지 않는다. 프로토 마진 구조상 **엣지가 검증되지 않은 상태의 베팅은 기대 손실**이다.
- 데이터 수집 시 각 사이트의 이용약관·robots.txt 를 준수한다.
- 문제성 도박이 우려되면 한국도박문제예방치유원 ☎ 1336.
