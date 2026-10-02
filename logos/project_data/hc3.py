import pandas as pd
from huggingface_hub import hf_hub_download
import re


def clean_hc3(df):
    error_pattern = (
        r"Too many requests in 1 hour|authentication token has expired|network error|"
        r"There was an error generating a response|ChatGPT Dec \d+ Version|Clear conversations"
    )
    # Remove text with HC3 specific API errors
    error = df["text"].str.contains(error_pattern, flags=re.IGNORECASE, na=False)
    return df[~error].drop_duplicates(subset="text").reset_index(drop=True)


def load_hc3():
    """
    Getting all project_data instead of domain specific
    will help the models generalise better,
    and for measuring domain shift
    """
    path = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename="all.jsonl",
        repo_type="dataset"
    )
    # Stream the .jsonl file as df
    raw = pd.read_json(path, lines=True)

    sources = [
        ("human_answers", 0, "human"),
        ("chatgpt_answers", 1, "chatgpt"),
    ]

    rows = []
    # Extract the human and ChatGPT answers and label them
    for index, row in raw.iterrows():
        for col, label, name in sources:
            for answer in row.get(col) or []:
                domain = row.get("source")

                if answer and answer.strip():
                    rows.append({
                        "text": answer,
                        "label": label,
                        "generator": name,
                        "family": name,
                        "domain": domain,
                        "question_id": index
                    })

    df = pd.DataFrame(
        rows,
        # The project_data schema to be used for the project
        columns=["text", "label", "generator", "family", "domain", "question_id"]
    )
    return df