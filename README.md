# xgboot-public-opinion-sentiment-stock-prediction
Training process and model only, excluding data preprocessing
Input items :[date,open,high,low,close,volume,sma_5,sma_10,sma_20,ema_12,ema_26,rsi_14,rsi_7,rsi_change,stddev,bb_middle,bb_upper,bb_lower,bb_width,bb_position,stoch_k,stoch_d,roc,mom,willr,price_to_sma20,price_change,price_change_pct,high_low_pct,body_pct,upper_shadow,lower_shadow,obv,volume_sma20,volume_ratio,stock_code,y,Sentiment_Mean,Sentiment_Std]

There are still many areas for improvement. Because the overall correlation is not significant, a very conservative strategy was adopted, aiming to maximize precision while ensuring the basic AUC and Recall.
