import pandas as pd

# 載入 CSV
ans = pd.read_csv("ad_legality_judgement_by_agent.csv")
query = pd.read_csv("final_project_query.csv")
# 保留 ID 與 Answer 欄位



mask = ans["Answer"] == 0.52

# 取得 B 的 description 欄位，根據 A 的條件
filtered_df = pd.DataFrame({
    "id": ans[mask].index,  # 加上行號作為 ID
    "Question": query.loc[mask, "Question"]
})
# 儲存成新 CSV
filtered_df.to_csv("test.csv", index=False)

