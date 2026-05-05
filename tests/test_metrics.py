                                                                                                    
import numpy as np
import pandas as pd                                                                                
from src.rl.metrics import calculate_all_metrics, sharpe_ratio, mdd
                                                                                                     
rng = np.random.default_rng(42)                                                                    
portfolio = pd.Series(rng.normal(0.0005, 0.015, 252))                                              
benchmark = pd.Series(rng.normal(0.0003, 0.010, 252))                                              
                  
print("=== 개별 함수 ===")                                                                         
print("샤프비율:", round(sharpe_ratio(portfolio), 4))
print("MDD     :", round(mdd(portfolio), 4))                                                       
                                                                                                     
print("\n=== calculate_all_metrics ===")
result = calculate_all_metrics(portfolio, benchmark)                                               
print(f"키 개수: {len(result)}  (12이어야 함)")
for k, v in result.items():                                                                        
    print(f"  {k:<25}: {v:.4f}")
                                    