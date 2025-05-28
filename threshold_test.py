import pandas as pd

THRESHOLD = 0.3

# 讀取原始 CSV
df = pd.read_csv("ad_legality_judgement_by_agent.csv")  # 修改為你的檔名

# 將 Answer 大於 THRESHOLD 設為 1，否則設為 0
df["Answer"] = (df["Answer"] > THRESHOLD).astype(int)

# 只保留 ID 和轉換後的 Answer 欄位
df = df[["ID", "Answer"]]

# 輸出為新的 CSV
df.to_csv("answer.csv", index=False)
