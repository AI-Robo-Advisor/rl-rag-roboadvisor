from src.agent.risk_tags import extract_rl_risk_tags, get_risk_vector, RL_RISK_TAGS                
                                                                                                     
  # 케이스 1: 규제변경 + 급등락                                                                      
text1 = "금융위원회가 가상자산 규제변경을 발표하자 시장이 급락했다"
tags1 = extract_rl_risk_tags(text1)                                                                
print("케이스1 태그:", tags1)                                                                      
print("케이스1 벡터:", get_risk_vector(tags1))  # [1. 0. 1.] 기대                                  
                                                                                                     
  # 케이스 2: 실적쇼크 + 급등락
text2 = "삼성전자 어닝쇼크로 주가 급락"                                                            
tags2 = extract_rl_risk_tags(text2)                                                                
print("케이스2 태그:", tags2)
print("케이스2 벡터:", get_risk_vector(tags2))  # [0. 1. 1.] 기대                                  
                                                                                                     
  # 케이스 3: 빈 텍스트
tags3 = extract_rl_risk_tags("")                                                                   
print("케이스3 태그:", tags3)
print("케이스3 벡터:", get_risk_vector(tags3))  # [0. 0. 0.] 기대                                  
   
  # 태그 순서 확인                                                                                   
print("태그 순서:", RL_RISK_TAGS)