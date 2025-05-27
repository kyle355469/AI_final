import pandas as pd

# 載入 CSV
df = pd.read_csv("ad_legality_judgement_by_agent.csv")

# 保留 ID 與 Answer 欄位
df_filtered = df[["ID", "Answer"]].copy()


# 儲存為新的檔案（選用）
df_filtered.to_csv("answer.csv", index=False)

