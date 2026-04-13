"""
Streamlit 6탭 대시보드 (구현 예정).

FastAPI HTTP 통신만 사용하며, 모델을 직접 로드하지 않습니다.

탭 구성 (Sprint 2~3 구현 예정):
  Tab 1: 포트폴리오 현황
  Tab 2: 투자 분석 (RAG 리포트)
  Tab 3: 리스크 모니터링
  Tab 4: 백테스트 결과
  Tab 5: RL 학습 현황
  Tab 6: 시장 국면 (ECOS 지표)
"""
import streamlit as st

st.set_page_config(page_title="AI Robo Advisor", layout="wide")
st.title("AI Robo Advisor Dashboard")
st.info("대시보드 구현 중입니다. (Sprint 2~3 예정)")
