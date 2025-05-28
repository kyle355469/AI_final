# AI_final
## intro
THis is the final project form AI 2025, NTU CSIE:
for project detail please look at [here](https://docs.google.com/presentation/d/1TrBFk5A7fMfh60rGPmu3xdWQ78Cg-IiHZi-g-EQ3oI4/edit?usp=sharing)
following is file description:
* v1/2/3&4.py : each version of final project's test result, see detail [here](https://docs.google.com/spreadsheets/d/1mRxqmu4xJbp-S1nUJGqFsXGhO7F7piEEUoOud6WOX68/edit?usp=sharing).
* threshold_test.py : to change probability into binary, with given threshold variable. 
* rule_build.py : for given .json files, turn them into three json file: law_text, legal_phrases and violation_phrase_map
* filter.pt : if have multiple keys for output, this file can help change into kaggle competion's format.
## Using step:
* pip install -r requirement.txt
* python3 v1/2/3&4.py
* python3 threshold_test.py
