# config.py — 프로젝트 공통 상수
# 담당: 이문정. 현재는 src/data/ 구현을 위한 최소 플레이스홀더.
# 추가 상수가 필요하면 이문정에게 PR로 요청.

TICKERS_GLOBAL: list[str] = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"]
TICKERS_KR: list[str]     = ["069500", "114260"]
START_DATE: str            = "2020-01-01"
END_DATE: str              = "2025-12-31"

DATA_RAW_DIR: str          = "data/raw/"
DATA_PROCESSED_DIR: str    = "data/processed/"
