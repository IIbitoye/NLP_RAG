import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_results_final.json")
CSV_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_grading_sheet_final.csv")

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)


df.to_csv(CSV_PATH, index=False)
print(f"Created grading sheet at {CSV_PATH}")