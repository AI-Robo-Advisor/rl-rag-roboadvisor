from src.agent.graph import run_graph                                                              
                  
result = run_graph("삼성전자 HBM 실적 전망은?")                                                    
  
print("=== sources ===")                                                                           
print(result.get("sources"))
                                                                                                     
print("\n=== risk_tags (3종) ===")
print(result.get("risk_tags"))                                                                     
                  
print("\n=== reasoning_trace ===")
print(result.get("reasoning_trace"))
                                                                                                     
print("\n=== report 미리보기 ===")
print(result.get("response", "")[:300])  