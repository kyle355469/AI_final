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

DB_PATH = "./final_project_db"

# ─── RAG ───
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

def build_vectordb():
    json_paths = [
        ("data/violation_phrase_map.json", "illegal")
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_docs = []

    for path, tag in json_paths:
        loader = JSONLoader(file_path=path, jq_schema=".[]", text_content=False)
        pages = loader.load()

        for page in pages:
            content = page.page_content
            metadata = {"source": tag}
            
            if len(content) > 300:  # 條文才切
                chunks = splitter.split_text(content)
                all_docs += [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
            else:
                all_docs.append(Document(page_content=content, metadata=metadata))

    vectordb = Chroma.from_documents(all_docs, embedding=embedding, persist_directory=DB_PATH)
    vectordb.persist()

def w_RAG(ad_type: str, keywords: list[str]) -> dict:
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})

    query_text = " ".join(keywords)
    docs = retriever.get_relevant_documents(query_text)

    result = {
        "violation_sample": [],
        "ad_type": ad_type
    }

    for doc in docs:
        source = doc.metadata.get("source", "")
        content = doc.page_content.strip()

        if source == "illegal":
            result["violation_sample"].append(content)
    print(result)
    return result

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

# FINDER_AGENT = build_agent(
#     "law_finder_agent",
#     "你是違規條文查找助理，輸入是一句廣告詞與其類型，請根據給予的違反關鍵字列出和本廣告相關的違規例句，並輸出為："
#     "{\"violation_sample\": [\"<sample 1>\", \"<sample 2>\", ...]}"
#     "不可以在輸出中提到 json"
# )

JUDGER_AGENT = build_agent(
    "judger_agent",
    "你是一位法律判定助理，根據 extract_agent 的輸出，請判斷此廣告是否違法，另外附上finder_agent 提供的已知違法關鍵字，並解釋原因"
)

SCORE_AGENT = build_agent(
    "score_agent",
    "你是一位收集助理，根據 judger_agent 的輸出，請輸出此廣告違法的機率，數字介於 0.00 ~ 1.00 之間。不要解釋，只提供數字"
)

ENTRY = build_agent("entry", "主控協調者")

# ─── CONVERSATION EXECUTOR ───
def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
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
                    print(f"Error parsing data: {past['content']}")
                    check = 1
                    continue
        elif step["recipient"] is EXTRACT_AGENT:
            try:
                ctx["keywords"] = json.loads(out).get("keywords", [])
            except:
                print(f"Error parsing keywords: {out}")
                
            if check:
                ctx["extract_ref"] = w_RAG(ad_type="unknown", keywords=ctx["keywords"])
            else:
                ctx["extract_ref"] = w_RAG(ad_type=ctx["ad_type"], keywords=ctx["keywords"])
                
        # elif step["recipient"] is FINDER_AGENT:
        #     ctx["is_legal"] = (json.loads(out))
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
                {"recipient": FETCH_AGENT, "message": "請判斷下列廣告類型：{ad_text}", "summary_method": "last_msg", "max_turns": 1},
                {"recipient": EXTRACT_AGENT, "message": "給我以下廣告的所有功效關鍵字：{ad_text}", "summary_method": "last_msg", "max_turns": 1},
                #{"recipient": FINDER_AGENT, "message": "輸入廣告：{ad_text}\n類型：{ad_type}, 比對用法關鍵字：{illegal_phrases}", "summary_method": "last_msg", "max_turns": 1},
                {"recipient": JUDGER_AGENT, "message": "{{\"關鍵字\": {keywords}}}, {{\"參考用\": {extract_ref}}}", "summary_method": "last_msg", "max_turns": 1},
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
            results.append({"query": str(int(row[0]) - 1), "Answer": 0})

    pd.DataFrame(results).to_csv("ad_legality_judgement_by_agent.csv", index=False)
    print("✅ 判定完成，結果輸出至 ad_legality_judgement_by_agent.csv")
    

if __name__ == "__main__":
    main()
