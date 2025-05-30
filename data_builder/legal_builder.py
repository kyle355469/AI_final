import json
import os
from pathlib import Path



file_list = [
    "食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json",
    "食品藥品健康食品得使用詞句補充案例.json"
]
legal_phrase = set()
legal_map = {}
for file in file_list:
    path = Path(__file__).resolve().parent.parent / "sample" / file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if file == "食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json":
            cases = data['categories']
            for case in cases:
                phrase = case.get('cases')
                if not phrase:
                    subcategories = case.get('subcategories', [])
                    for sub in subcategories:
                        cat_type = sub.get('cases')
                        for sub_case in cat_type:
                            legal_map[sub_case['ad_content'].strip()] = sub_case['description'].strip() 
                            legal_phrase.add(sub_case['ad_content'])
                elif isinstance(phrase, list):
                    for p in phrase:
                        legal_map[p['ad_content'].strip()] = "合法用語"
                        legal_phrase.add(p['ad_content'])
                else:
                    legal_map[phrase.strip()] = "合法用語"
                    legal_phrase.add(phrase.strip())
        if file == "食品藥品健康食品得使用詞句補充案例.json":
            cases = data['cases']
            for case in cases:
                legal_map[case['content'].strip()] = "合法用語"
                legal_phrase.add(case['content'])
                
        
with open(Path(__file__).resolve().parent.parent / "data" / "legal_phrases_map.json", "w", encoding="utf-8") as f:
    json.dump(legal_map, f, ensure_ascii=False, indent=2)
    
with open(Path(__file__).resolve().parent.parent / "data" / "legal_phrases.json", "w", encoding="utf-8") as f:
    json.dump(sorted(list(legal_phrase)), f, ensure_ascii=False, indent=2)