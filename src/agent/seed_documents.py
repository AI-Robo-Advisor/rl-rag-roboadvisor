"""Local seed documents for deterministic RAG serving readiness."""

from __future__ import annotations

from src.agent.vectorstore import collection_document_count, upsert_documents

SEED_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "seed_spy_qqq_rates_2026",
        "title": "미국 주식 ETF와 금리 민감도",
        "summary": (
            "SPY는 S&P 500 대형주, QQQ는 나스닥 100 성장주 노출을 제공한다. "
            "금리 인하 기대는 성장주 밸류에이션에 우호적일 수 있지만 경기 둔화가 "
            "동반되면 이익 전망 하향과 변동성 확대가 포트폴리오 리스크가 된다."
        ),
        "url": "https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy",
        "date": "2026-05-15",
        "category": "증시",
        "risk_label": "급등락",
    },
    {
        "id": "seed_tlt_114260_duration_2026",
        "title": "장기채와 단기 국고채 ETF의 듀레이션 리스크",
        "summary": (
            "TLT는 미국 장기채 ETF로 금리 하락 시 가격 상승 민감도가 크고, "
            "114260은 KODEX 국고채3년으로 상대적으로 듀레이션이 짧다. "
            "금리 경로가 불확실할 때 두 채권 ETF의 만기 차이는 방어 자산 내 리스크 배분에 중요하다."
        ),
        "url": "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf",
        "date": "2026-05-15",
        "category": "정책",
        "risk_label": "금리인상,급등락",
    },
    {
        "id": "seed_gld_vnq_inflation_2026",
        "title": "금과 리츠 ETF의 인플레이션 및 실물자산 노출",
        "summary": (
            "GLD는 금 가격에 연동되어 인플레이션과 달러 약세 환경에서 방어적 역할을 할 수 있다. "
            "VNQ는 미국 리츠 ETF로 부동산 현금흐름에 노출되며 금리 상승기에는 조달비용과 할인율 부담을 받는다."
        ),
        "url": "https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-gold-shares-gld",
        "date": "2026-05-15",
        "category": "증시",
        "risk_label": "급등락",
    },
    {
        "id": "seed_efa_eem_global_2026",
        "title": "선진국 및 신흥국 ETF의 글로벌 분산 효과",
        "summary": (
            "EFA는 미국 외 선진국 주식, EEM은 신흥국 주식 노출을 제공한다. "
            "달러 강세, 중국 경기, 지정학 리스크는 EEM 변동성을 키울 수 있고, "
            "글로벌 분산은 미국 주식 집중도를 낮추는 데 활용된다."
        ),
        "url": "https://www.ishares.com/us/products/239623/ishares-msci-emerging-markets-etf",
        "date": "2026-05-15",
        "category": "증시",
        "risk_label": "규제변경,급등락",
    },
    {
        "id": "seed_069500_kospi_2026",
        "title": "KODEX 200과 한국 주식시장 노출",
        "summary": (
            "069500 KODEX 200은 KOSPI 200 지수를 추종하는 한국 대표 주식 ETF다. "
            "반도체 수출, 원화 환율, 국내 금리, 외국인 수급 변화가 한국 주식 비중 조정의 주요 리스크 요인이다."
        ),
        "url": "https://www.samsungfund.com/etf/product/view.do?id=2ETF01",
        "date": "2026-05-15",
        "category": "증시",
        "risk_label": "실적쇼크,급등락",
    },
]


def ensure_seed_documents(persist_dir: str) -> int:
    """Seed deterministic RAG documents when the local Chroma collection is empty."""
    if collection_document_count(persist_dir) > 0:
        return 0

    documents = [item["summary"] for item in SEED_DOCUMENTS]
    metadatas = [
        {
            "title": item["title"],
            "summary": item["summary"],
            "url": item["url"],
            "date": item["date"],
            "category": item["category"],
            "risk_label": item["risk_label"],
            "source": "local_seed",
        }
        for item in SEED_DOCUMENTS
    ]
    ids = [item["id"] for item in SEED_DOCUMENTS]
    upsert_documents(documents, metadatas, ids, persist_dir=persist_dir)
    return len(ids)
