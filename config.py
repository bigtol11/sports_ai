"""
config.py
---------
프로젝트 전역 설정. 경로 / 종목별 스펙(SPORTS) / 모델 하이퍼파라미터 / 자금관리 / 토토 게임 유형을 한곳에서 관리한다.
다른 모듈은 이 파일의 값만 참조하고 하드코딩하지 않는다. (종목 추가 = SPORTS 에 항목 추가 + scraper 등록)
"""
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 경로
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # 원본 캐시
PROCESSED_DIR = DATA_DIR / "processed"  # 종목별 표준 스키마 데이터: {sport}_matches.csv
INPUT_DIR = DATA_DIR / "input"          # 사용자가 넣는 배당/회차 CSV
MODEL_DIR = BASE_DIR / "models"         # 종목별 모델: {sport}.joblib
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"             # 베팅/예측 기록 (자가진화 파이프라인 입력)

for _d in (RAW_DIR, PROCESSED_DIR, INPUT_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def matches_file(sport: str) -> Path:
    return PROCESSED_DIR / f"{sport}_matches.csv"


def model_file(sport: str) -> Path:
    return MODEL_DIR / f"{sport}.joblib"


# ══════════════════════════════════════════════════════════════════════
# 수집 공통
# ══════════════════════════════════════════════════════════════════════
REQUEST_TIMEOUT = 20      # 초
REQUEST_DELAY = 1.0       # 요청 간 대기(초) - 서버 부하 방지
USER_AGENT = "Mozilla/5.0 (sports_ai research)"

# ══════════════════════════════════════════════════════════════════════
# 종목 스펙
#   outcomes        : 모델 분류 클래스 (H=홈승, D=무, A=원정승)
#   score_dist      : 몬테카를로 점수 분포 (poisson / negbin / normal / sets)
#   points          : 폼 계산용 (승, 무, 패) 점수
#   elo_k / elo_home_adv / mov_scale : Elo 갱신 계수 / 홈 어드밴티지 / 점수차 가중 스케일
#   rolling_stats   : 경기 후 기록되는 스탯 → 팀별 최근 N경기 평균 피처 (컬럼명 home_<stat>, away_<stat>)
#   prematch        : 경기 전에 알 수 있는 값 → 그대로 피처 (컬럼명 home_<p>, away_<p>)
# ══════════════════════════════════════════════════════════════════════
SPORTS = {
    "soccer": {
        "name_kr": "축구",
        "leagues": {
            "E0": "England Premier League", "SP1": "Spain La Liga", "D1": "Germany Bundesliga",
            "I1": "Italy Serie A", "F1": "France Ligue 1",
            "E1": "England Championship", "N1": "Netherlands Eredivisie", "P1": "Portugal Primeira Liga",
            "KLEAGUE": "K League 1 (스켈레톤)",
        },
        "seasons": ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"],
        "outcomes": ["H", "D", "A"],
        "score_dist": "poisson",
        "points": (3, 1, 0),
        "elo_k": 20.0, "elo_home_adv": 65.0, "mov_scale": 1.0,
        # draw / total 은 model.py 에서 파생 생성 (무승부 성향 변수)
        "rolling_stats": ["shots", "sot", "corners", "xg", "draw", "total"],
        "prematch": ["injury_ratio"],
    },
    "baseball": {
        "name_kr": "야구",
        "leagues": {"MLB": "Major League Baseball", "KBO": "KBO League (스켈레톤)"},
        "seasons": ["2022", "2023", "2024", "2025", "2026"],
        "outcomes": ["H", "A"],
        "score_dist": "negbin",
        "nb_dispersion": 4.0,                       # 득점 분포 과산포 파라미터
        "max_extra_innings": {"MLB": 20, "KBO": 2},  # KBO 연장 규정은 시즌마다 확인
        "points": (1, 0.5, 0),
        "elo_k": 6.0, "elo_home_adv": 24.0, "mov_scale": 2.0,
        "rolling_stats": ["hits", "errors", "bullpen_ip"],
        "prematch": ["sp_era", "sp_whip", "sp_fip", "sp_k9", "sp_hand", "bullpen_era", "injury_ratio"],
    },
    "basketball": {
        "name_kr": "농구",
        "leagues": {"NBA": "National Basketball Association (스켈레톤)", "KBL": "KBL (스켈레톤)"},
        "seasons": ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"],
        "outcomes": ["H", "A"],
        "score_dist": "normal",
        "score_std": 11.5,                          # 팀 득점 표준편차
        "points": (1, 0.5, 0),
        "elo_k": 20.0, "elo_home_adv": 70.0, "mov_scale": 8.0,
        "rolling_stats": ["pace", "ortg", "drtg", "q1", "q2", "q3", "q4", "fg3_rate"],
        "prematch": ["injury_ratio"],
    },
    "volleyball": {
        "name_kr": "배구",
        "leagues": {"VM": "V-League 남자 (스켈레톤)", "VW": "V-League 여자 (스켈레톤)"},
        "seasons": ["2223", "2324", "2425", "2526", "2627"],
        "outcomes": ["H", "A"],
        "score_dist": "sets",                       # 5세트 3선승
        "points": (1, 0.5, 0),
        "elo_k": 20.0, "elo_home_adv": 40.0, "mov_scale": 1.0,
        "rolling_stats": ["attack_eff", "receive_eff", "blocks", "aces", "points_total"],
        "prematch": ["injury_ratio"],
    },
}

FORM_WINDOW = 6           # 최근 N경기 폼
ELO_INIT = 1500.0

# ══════════════════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════════════════
TEST_RATIO = 0.15         # 시간순 마지막 15%를 검증용으로 사용
MIN_TRAIN_ROWS = 300      # 이보다 적으면 경고
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8,
}
LGBM_PARAMS = {
    "n_estimators": 400, "num_leaves": 15, "learning_rate": 0.03,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "verbose": -1,
}
ENSEMBLE_WEIGHTS = (0.5, 0.5)   # (XGB, LGBM)

MC_SIMULATIONS = 10_000         # 몬테카를로 가상 경기 수

# ══════════════════════════════════════════════════════════════════════
# 프로토 (EV / 켈리) — "다폴더 억제, 1~2폴더 집중" 원칙
# ══════════════════════════════════════════════════════════════════════
BANKROLL = 100_000          # 총 운용 자금(원)
KELLY_FRACTION = 0.25       # 풀켈리의 1/4 (변동성 완화)
MAX_STAKE_RATIO = 0.05      # 1회 베팅 최대 비중 (자금의 5%)
MAX_DAILY_EXPOSURE = 0.15   # 하루 총 베팅 상한 (자금의 15%)
MIN_EV = 0.03               # 기대가치 3% 이상만 추천
MIN_PROB = 0.05             # 극단 저확률 제외
MIN_STAKE = 100             # 최소 구매 단위(원)
PROTO_MIN_FOLDERS = 1       # 베트맨 규정상 단폴 불가 경기면 2로 설정
PROTO_MAX_FOLDERS = 2       # 원칙: 최대 2폴더

# ══════════════════════════════════════════════════════════════════════
# 토토 게임 유형
#   rule: wdl(승무패) / margin(승X패: |점수차|<=margin 이면 가운데) / score_bucket(팀별 득점 구간) / set_score(세트스코어)
#   ※ 승1패·승5패·스페셜 구간과 경기 수는 회차 공지마다 다를 수 있으므로 베트맨 공지로 반드시 확인
# ══════════════════════════════════════════════════════════════════════
TOTO_GAME_TYPES = {
    "soccer_wdl":       {"sport": "soccer",     "rule": "wdl",    "labels": ["승", "무", "패"], "n_games": 14},
    "baseball_w1l":     {"sport": "baseball",   "rule": "margin", "margin": 1, "labels": ["승", "1", "패"], "n_games": 14},
    "basketball_w5l":   {"sport": "basketball", "rule": "margin", "margin": 5, "labels": ["승", "5", "패"], "n_games": 14},
    "soccer_special":   {"sport": "soccer",     "rule": "score_bucket", "buckets": [(0, 0), (1, 1), (2, 2), (3, 99)],
                         "labels": ["0", "1", "2", "3+"], "n_games": None},
    "baseball_special": {"sport": "baseball",   "rule": "score_bucket", "buckets": [(0, 2), (3, 4), (5, 6), (7, 99)],
                         "labels": ["0-2", "3-4", "5-6", "7+"], "n_games": None},
    # 농구 스페셜 구간은 KBL 기준 예시 - NBA 회차는 공지된 구간으로 교체
    "basketball_special": {"sport": "basketball", "rule": "score_bucket",
                           "buckets": [(0, 69), (70, 79), (80, 89), (90, 999)],
                           "labels": ["~69", "70-79", "80-89", "90+"], "n_games": None},
    "volleyball_match": {"sport": "volleyball", "rule": "set_score",
                         "labels": ["3:0", "3:1", "3:2", "2:3", "1:3", "0:3"], "n_games": None},
}
TOTO_COST_PER_COMBINATION = 100   # 1조합당 금액(원) - 실제 판매 규정에 맞게 조정
TOTO_DEFAULT_BUDGET = 9_600       # 기본 예산(원) → 최대 조합 수 = 예산 / 조합당 금액

# 한글/베트맨 표기 → 데이터 소스 영문 팀명 (종목 공통, 필요한 만큼 추가)
TEAM_ALIASES = {
    "맨체스터U": "Man United", "맨체스터C": "Man City", "토트넘": "Tottenham", "아스널": "Arsenal",
    "리버풀": "Liverpool", "첼시": "Chelsea", "뉴캐슬": "Newcastle", "애스턴빌라": "Aston Villa",
    "LA다저스": "Los Angeles Dodgers", "뉴욕양키스": "New York Yankees",
}
