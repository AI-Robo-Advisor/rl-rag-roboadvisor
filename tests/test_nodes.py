from src.agent.nodes import analyst_node
                                                                                                     
state = {       
  "query": "삼성전자 반도체 전망은?",                                                            
  "context": "[문서 1] 삼성전자 HBM 수요 급증\n날짜: 2025-01-01\n본문: HBM 공급 부족으로 실적 개선 기대\n출처: https://example.com/news/1",                                                      
  "documents": [{"content": "HBM 수요 급증 실적쇼크 우려", "metadata": {"url": "https://example.com/news/1"}}],                                                                   
  "risk_tags": ["변동성_리스크"],
  "messages": ["[THINK][planner] 플랜 완료", 
  "[THINK][researcher] 검색 완료"],                   
  "retry_count": 0,                                                                              
  "needs_research_retry": False,                                                                 
  "sources": [],                                                                                 
  "reasoning_trace": "",                                                                         
  }               
                                                                                                     
result = analyst_node(state)

print("=== sources ===")
print(result["sources"])
                                                                                                     
print("\n=== risk_tags (3종) ===")
print(result["risk_tags"])                                                                         
                                                                                                     
print("\n=== reasoning_trace ===")
print(result["reasoning_trace"])                                                                   
                  
print("\n=== report 미리보기 ===")
print(result["response"][:300])