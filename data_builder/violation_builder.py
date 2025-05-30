import json
import os
from pathlib import Path


OUT_DIR = Path("../data")

file_list = [
    "衛生福利部暨臺北市政府衛生局食品廣告例句209.json",
    "衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json",
    "中藥成藥不適當共通性廣告詞句.json",
    "化妝品涉及影響生理機能或改變身體結構之詞句.json",
    "中醫藥司之中藥成藥效能、適應症語意解析及中藥廣告違規態樣釋例彙編.json",
    "13項保健功效及不適當功效延申例句之參考.json",
]
violation_phrase = set()
violation_map = {}
for file in file_list:
    path = Path(__file__).resolve().parent.parent / "sample" / file
    if not path.exists():
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if file == "衛生福利部暨臺北市政府衛生局食品廣告例句209.json":
            cases = data['advertisement_violation_cases']['cases']
            for case in cases:
                reason = case['violation_type']
                content = case['ad_content']
                violation_map[content.strip()] = reason
                violation_phrase.add(content)
        if file == "衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json":
            cases = data['health_food_violations']['cases']
            for case in cases:
                reason = case['violation_type']
                content = case['ad_content']
                violation_map[content.strip()] = reason
                violation_phrase.add(content)
        if file == "中藥成藥不適當共通性廣告詞句.json":
            cases = data['inappropriate_tcm_ads']['violation_categories']
            for case in cases:
                reason = case['category']
                contents = case['prohibited_terms']
                for content in contents:
                    violation_map[content.strip()] = reason
                    violation_phrase.add(content)
        if file == "化妝品涉及影響生理機能或改變身體結構之詞句.json":
            cases = data['prohibited_physiological_claims']['categories']
            for case in cases:
                reason = case['category']
                contents = case['prohibited_terms']
                for content in contents:
                    violation_map[content.strip()] = reason
                    violation_phrase.add(content)
        if file == "中醫藥司之中藥成藥效能、適應症語意解析及中藥廣告違規態樣釋例彙編.json":
            cases = data['categories']
            for case in cases:
                reason = case['category']
                contents = case['examples']
                for content in contents:
                    violation_map[content['ad_content'].strip()] = reason
                    violation_phrase.add(content['ad_content'])
        if file == "13項保健功效及不適當功效延申例句之參考.json":
            cases = data['health_function_guidelines']['items']
            for case in cases:
                reason = case['function']
                contents = case['inappropriate_claims']
                for content in contents:
                    violation_map[content.strip()] = reason
                    violation_phrase.add(content)
        # print(violation_map)
        
with open(Path(__file__).resolve().parent.parent / "data" / "illegal_phrases_map.json", "w", encoding="utf-8") as f:
    json.dump(violation_map, f, ensure_ascii=False, indent=2)
    
with open(Path(__file__).resolve().parent.parent / "data" / "illegal_phrases.json", "w", encoding="utf-8") as f:
    json.dump(sorted(list(violation_phrase)), f, ensure_ascii=False, indent=2)