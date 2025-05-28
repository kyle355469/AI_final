import json
import os
from pathlib import Path

# 原始檔案資料夾
SOURCE_DIR = Path("sample")

# 輸出資料夾
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# === 1. 違規語句收集 ===
violation_phrase_map = {}
violation_sources = [
    "衛生福利部暨臺北市政府衛生局食品廣告例句209.json",
    "衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json",
    "中藥成藥不適當共通性廣告詞句.json",
    "化妝品涉及影響生理機能或改變身體結構之詞句.json",
]
illegal_phrases = set()
for file in violation_sources:
    path = SOURCE_DIR / file
    if not path.exists():
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for key in data:
            if isinstance(data[key], dict) and "cases" in data[key]:
                for case in data[key]["cases"]:
                    phrase = case.get("ad_content") or case.get("content")
                    #reason = case.get("violation_type") or "違規但無分類"
                    if phrase:
                        illegal_phrases.add(phrase.strip())
                        # violation_phrase_map[phrase.strip()] = reason

# === 2. 法條整理 ===
law_texts = set()
law_sources = [
    "食品安全衛生管理法.json",
    "化粧品衛生管理法.json",
    "食品、化粧品、藥物、醫療器材相關法規彙編.json",
]
for file in law_sources:
    path = SOURCE_DIR / file
    if not path.exists():
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for category in data.get("categories", []):
            for subcategory in category.get("subcategories", []):
                for case in subcategory.get("cases", []):
                    content = case.get("content")
                    if content:
                        law_texts.add(content.strip())
            for case in category.get("cases", []):
                content = case.get("content") or case.get("內容")
                if content:
                    law_texts.add(content.strip())
                    

# === 3. 合法用語 ===
legal_phrases = set()
legal_path = SOURCE_DIR / "食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json"
if legal_path.exists():
    with open(legal_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for category in data.get("categories", []):
            for subcategory in category.get("subcategories", []):
                for case in subcategory.get("cases", []):
                    phrase = case.get("ad_content")
                    if phrase:
                        legal_phrases.add(phrase.strip())
            for case in category.get("cases", []):
                phrase = case.get("ad_content")
                if phrase:
                    legal_phrases.add(phrase.strip())

# === 4. 儲存三個結果 ===
with open(OUTPUT_DIR / "violation_phrase_map.json", "w", encoding="utf-8") as f:
    #json.dump(violation_phrase_map, f, ensure_ascii=False, indent=2)
    json.dump( sorted(list(illegal_phrases)), f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "law_texts.json", "w", encoding="utf-8") as f:
    json.dump(sorted(list(law_texts)), f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "legal_phrases.json", "w", encoding="utf-8") as f:
    json.dump(sorted(list(legal_phrases)), f, ensure_ascii=False, indent=2)

print("✅ 整理完成，三個檔案已儲存在 ./data/")
