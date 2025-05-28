## v3 & v4

from __future__ import annotations
from autogen import ConversableAgent
import os, sys, json, ast, pandas as pd
from typing import List
# ─── API KEY SETUP ───
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("\u2757 Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ─── DATA SETUP ───
with open("data/violation_phrase_map.json", "r", encoding="utf-8") as f:
    VIOLATION_PHRASE_MAP = json.load(f)
with open("data/legal_phrases.json", "r", encoding="utf-8") as f:
    LEGAL_PHRASES = json.load(f)
with open("data/law_texts.json", "r", encoding="utf-8") as f:
    LAW_TEXT = json.load(f)

# ─── Utility Agent Factory ───
def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

# ─── AGENT SETUP ───
# FETCH_AGENT = build_agent(
#     "data_fetch_agent",
#     "你是一位廣告分析助理，請判斷輸入廣告文字是屬於哪一類（食品、健康食品、化妝品、中藥成藥、醫療器材、其他），並固定以下列格式輸出：{\"ad_type\"：\"<類型>\"}"
#     #"請注意<類型>外面不能使用引號，雙引號"
# )

EXTRACT_AGENT = build_agent(
    "extract_agent",
    "你是一位廣告關鍵字提取助理，請從輸入的廣告文字中只提取關於功效關鍵字，並輸出為：{\"keywords\": [\"<關鍵字1>\", \"<關鍵字2>\", ...]}"
)

## 換成 REG
FINDER_AGENT = build_agent(
    "law_finder_agent",
    "你是違規條文查找助理，輸入是一句廣告詞與其類型，請從提供的合法/違法關鍵字中列出和廣告關鍵字有相關的違法/合法關鍵詞，並輸出為："
    "{\"violation_sample\": [\"<sample 1>\", \"<sample 2>\", ...], \"legal_sample\": [\"<legal sample 1>\", \"<legal sample 2>\", ...], \"ad_type\": \"<類型>\"}"
    "不可以在輸出中提到 json"
)

JUDGER_AGENT = build_agent(
    "judger_agent",
    "你是一位法律判定助理，根據 finder_agent 提供的資料和條文，請判斷此廣告是否違法，並解釋原因"
)

SCORE_AGENT = build_agent(
    "score_agent",
    "你是一位收集助理，根據 judger_agent 的輸出，請輸出此廣告違法的機率，數字介於 0.00 ~ 1.00 之間。不要解釋，只提供數字"
)

ENTRY = build_agent("entry", "主控協調者")

# ─── CONVERSATION EXECUTOR ───
def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    ctx["illegal_phrases"] = VIOLATION_PHRASE_MAP
    ctx["legal_phrases"] = LEGAL_PHRASES
    ctx["law_text"] = LAW_TEXT
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        # if step["recipient"] is FETCH_AGENT:
        #     for past in reversed(chat.chat_history):
        #         try:
        #             data = ast.literal_eval(past['content'])
        #             print(data["ad_type"])
        #             ctx["ad_type"] = data["ad_type"]
        #             break
        #         except:
        #             print(f"Error parsing data: {past['content']}")
        #             continue
        if step["recipient"] is EXTRACT_AGENT:
            try:
                ctx["keywords"] = json.loads(out).get("keywords", [])
            except:
                print(f"Error parsing keywords: {out}")
                exit()
        elif step["recipient"] is FINDER_AGENT:
            ctx["extract_ref"] = (json.loads(out))
        elif step["recipient"] is JUDGER_AGENT:
            ctx["final_judgement"] = out
        elif step["recipient"] is SCORE_AGENT:
            ctx["final_judgement"] = out.strip()
    return ctx


ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ─── MAIN ENTRY ───
def main():
    #df = pd.read_csv("final_project_query.csv")
    df = pd.read_csv("test.csv")
    results: List[dict] = []

    for key, row in df.iterrows():
        
        ENTRY._initiate_chats_ctx = {"ad_text": str(row[1]).strip()}
        try:
            chat_sequence = [
                # {"recipient": FETCH_AGENT, "message": "請判斷下列廣告類型：{ad_text}", "summary_method": "last_msg", "max_turns": 1},
                {"recipient": EXTRACT_AGENT, "message": "給我以下廣告的所有功效關鍵字：{ad_text}", "summary_method": "last_msg", "max_turns": 1},
                {"recipient": FINDER_AGENT, "message": "輸入廣告關鍵字：{keywords}\n, 參考用違法關鍵字：{illegal_phrases}, 參考用合法關鍵字：{legal_phrases}", "summary_method": "last_msg", "max_turns": 1},
                {"recipient": JUDGER_AGENT, "message": "{{\"廣告關鍵字\": {keywords}}}, {{\"條文\": {law_text}}}, {{\"參考用\": {extract_ref}}}", "summary_method": "last_msg", "max_turns": 1},
                #{"recipient": JUDGER_AGENT, "message": "{{\"廣告關鍵字\": {keywords}}}, 參考用違法字：{illegal_phrases}, 參考用合法字：{legal_phrases}", "summary_method": "last_msg", "max_turns": 1},
                {"recipient": SCORE_AGENT, "message": "{{\"is_legal\": {final_judgement}}}", "summary_method": "last_msg", "max_turns": 1}
            ]
            res = ENTRY.initiate_chats(chat_sequence)
            results.append({
                "ID": str(int(row[0]) - 1),
                # "ad_type": res.get("ad_type", ""),
                "Answer": str(res["final_judgement"]),
                # "matched_phrase": res.get("matched_phrase", ""),
                # "violation_reason": res.get("violation_reason", ""),
                # "law_reference": res.get("law_reference", "")
            })
        except Exception as e:
            print("error: ", e)
            results.append({"ID": str(int(row[0]) - 1), "Answer": 0})

    pd.DataFrame(results).to_csv("ad_legality_judgement_by_agent.csv", index=False)
    print("✅ 判定完成，結果輸出至 ad_legality_judgement_by_agent.csv")
    

if __name__ == "__main__":
    main()
