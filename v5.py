## v5

from __future__ import annotations
from autogen import ConversableAgent
import os, sys, json, ast, pandas as pd
from typing import List
import difflib
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
import openai

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
DB_PATH = "./final_project_db"

# ─── RAG ───
# def retrieve_related_context(keywords: list[str]) -> dict:
#     def match_keywords(word_list, keywords):
#         matched = set()
#         for kw in keywords:
#             close_matches = difflib.get_close_matches(kw, word_list, n=5, cutoff=0.6)
#             matched.update(close_matches)
#         return list(matched)

#     # 1. 找出相關違法 / 合法詞條
#     matched_violation = match_keywords(VIOLATION_PHRASE_MAP.keys(), keywords)
#     matched_legal = match_keywords(LEGAL_PHRASES.keys(), keywords)

#     # 2. 擷取違法樣例
#     violation_samples = [VIOLATION_PHRASE_MAP[k] for k in matched_violation]
#     legal_samples = [LEGAL_PHRASES[k] for k in matched_legal]

#     # 3. 擷取相關法條（所有條文都抓）
#     law_refs = [f"{law['法條']}：{law['內容']}" for law in LAW_TEXT]

#     return {
#         "violation_sample": sum(violation_samples, []),  # flatten
#         "legal_sample": sum(legal_samples, []),
#         "law": law_refs
#     }

# def finder_agent_with_rag(ad_type: str, keywords: list[str]) -> dict:
#     context = retrieve_related_context(keywords)
#     prompt = (
#         "你是條文查找助理，請根據以下資訊列出相關內容：\n\n"
#         f"【廣告類型】：{ad_type}\n"
#         f"【關鍵字】：{', '.join(keywords)}\n\n"
#         f"【違法樣例】：{context['violation_sample'][:5]}\n"
#         f"【合法樣例】：{context['legal_sample'][:5]}\n"
#         f"【法條】：{context['law'][:3]}\n\n"
#         "請輸出 JSON 格式：{\"violation_sample\": [...], \"legal_sample\": [...], \"law\": [...], \"ad_type\": \"...\"}。\n"
#         "不要在輸出中提到 JSON 這個詞。"
#     )

#     chat = ENTRY.initiate_chat(
#         ConversableAgent("finder_agent", system_message="你是一位知識輔助的 RAG 法律助理", llm_config=LLM_CFG),
#         message=prompt,
#         summary_method="last_msg",
#         max_turns=1
#     )
#     return json.loads(chat.summary)

def w_RAG(ad_type: str, keywords: list[str]) -> dict:
    # ======== Step 1: Load and Split JSON files =========
    json_paths = [
        ("data/law.json", "law"),
        ("data/legal.json", "legal"),
        ("data/illegal.json", "illegal")
    ]
    all_docs = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for path in json_paths:
        loader = JSONLoader(file_path=path, jq_schema=".[]", text_content=False)  # 讀取 JSON 每一筆
        pages = loader.load()
        pages = [
            Document(page.page_content, metadata={"source": tag})
            for page in pages
        ]
        docs = splitter.split_documents(pages)
        all_docs.extend(docs)

    # ======== Step 2: Embedding with OpenAI =========
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(documents=all_docs, embedding=embedding, persist_directory=DB_PATH)
    retriever = vectordb.as_retriever()

    # ======== Step 3: RAG Chain =========
    system_prompt = (
        "請根據下列 context 回答問題。"
        "你是條文查找助理，輸入是一句廣告詞與其類型，請從提供的合法/違法關鍵字中列出和廣告關鍵字有相關的違法/合法關鍵詞與相關的條文，並輸出為："
        "{{\"violation_sample\": [\"<sample 1>\", \"<sample 2>\", ...], \"legal_sample\": [\"<legal sample 1>\", \"<legal sample 2>\", ...], \"law\": [\"<law 1>\", \"<law 2>\", ...], \"ad_type\": \"<類型>\"}}"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

    # ======== Step 4: 提問 =========
    QUERY = f"本產品聲稱: {keywords}，請提供相關關鍵字和條文"
    relevant_docs = chain.invoke({"input": QUERY})["context"]  # context 文件會包含 metadata

    violation_sample = []
    legal_sample = []
    law_sample = []

    for doc in relevant_docs:
        content = doc.page_content.strip()
        source = doc.metadata.get("source", "")
        
        if source == "illegal":
            violation_sample.append(content)
        elif source == "legal":
            legal_sample.append(content)
        elif source == "law":
            law_sample.append(content)
    print(violation_sample, legal_sample, law_sample)

# ─── Utility Agent Factory ───
def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

# ─── AGENT SETUP ───
FETCH_AGENT = build_agent(
    "data_fetch_agent",
    "你是一位廣告分析助理，請判斷輸入廣告文字是屬於哪一類（食品、健康食品、化妝品、中藥成藥、醫療器材、其他），並固定以下列格式輸出：{\"ad_type\"：\"<類型>\"}"
    #"請注意<類型>外面不能使用引號，雙引號"
)

EXTRACT_AGENT = build_agent(
    "extract_agent",
    "你是一位廣告關鍵字提取助理，請從輸入的廣告文字中只提取關於功效關鍵字，並輸出為：{\"keywords\": [\"<關鍵字1>\", \"<關鍵字2>\", ...]}"
)

## 換成 RAG
# FINDER_AGENT = build_agent(
#     "law_finder_agent",
#     "你是條文查找助理，輸入是一句廣告詞與其類型，請從提供的合法/違法關鍵字中列出和廣告關鍵字有相關的違法/合法關鍵詞與相關的條文，並輸出為："
#     "{\"violation_sample\": [\"<sample 1>\", \"<sample 2>\", ...], \"legal_sample\": [\"<legal sample 1>\", \"<legal sample 2>\", ...], \"law\": [\"<law 1>\", \"<law 2>\", ...], \"ad_type\": \"<類型>\"}"
#     "不可以在輸出中提到 json"
# )

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
    check = 0
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is FETCH_AGENT:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past['content'])
                    print(data["ad_type"])
                    ctx["ad_type"] = data["ad_type"]
                    break
                except:
                    check = 1
                    print(f"Error parsing data: {past['content']}")
                    continue
        elif step["recipient"] is EXTRACT_AGENT:
            try:
                ctx["keywords"] = json.loads(out).get("keywords", [])
            except:
                print(f"Error parsing keywords: {out}")
                exit()

            if check:
                ctx["extract_ref"] = w_RAG(ad_type="unknown", keywords=ctx["keywords"])
            else:
                ctx["extract_ref"] = w_RAG(ad_type=ctx["ad_type"], keywords=ctx["keywords"])

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
        # try:
        chat_sequence = [
            {"recipient": FETCH_AGENT, "message": "請判斷下列廣告類型：{ad_text}", "summary_method": "last_msg", "max_turns": 1},
            {"recipient": EXTRACT_AGENT, "message": "給我以下廣告的所有功效關鍵字：{ad_text}", "summary_method": "last_msg", "max_turns": 1},
            #{"recipient": FINDER_AGENT, "message": "輸入廣告關鍵字：{keywords}\n, 參考用違法關鍵字：{illegal_phrases}, 參考用合法關鍵字：{legal_phrases}, {{\"條文\": {law_text}}}", "summary_method": "last_msg", "max_turns": 1},
            {"recipient": JUDGER_AGENT, "message": "{{\"廣告關鍵字\": {keywords}}},  {{\"參考用\": {extract_ref}}}", "summary_method": "last_msg", "max_turns": 1},
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
        # except Exception as e:
        #     print("error: ", e)
        #     results.append({"ID": str(int(row[0]) - 1), "Answer": 0})

    pd.DataFrame(results).to_csv("ad_legality_judgement_by_agent.csv", index=False)
    print("✅ 判定完成，結果輸出至 ad_legality_judgement_by_agent.csv")
    

if __name__ == "__main__":
    main()
