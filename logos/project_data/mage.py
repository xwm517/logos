import pandas as pd
from datasets import load_dataset

from logos.project_data.preprocess import preprocess

main_domains = ["yelp", "eli5", "squad", "xsum", "wp", "roct", "hswag", "cmv", "sci_gen", "tldr"]

families = {
    "OpenAI GPT": ["gpt-3.5-trubo", "text-davinci-002", "text-davinci-003"],
    "LLaMA": ["7B", "13B", "30B", "65B"],
    "GLM-130B": ["GLM130B"],
    "FLAN-T5": ["flan_t5_small", "flan_t5_base", "flan_t5_large", "flan_t5_xl", "flan_t5_xxl"],
    "OPT": ["opt_125m", "opt_350m", "opt_1.3b", "opt_2.7b", "opt_6.7b", "opt_13b", "opt_30b", "opt_iml_max_1.3b", "opt_iml_30b"],
    "BigScience": ["t0_3b", "t0_11b", "bloom_7b"],
    "EleutherAI": ["gpt_j", "gpt_neox"],
}


def load_mage():
    mage = load_dataset("yaful/MAGE")
    # Stack the three splits into one table, remembering where each text came from
    splits = [mage[split].to_pandas().assign(split=split) for split in ["train", "validation", "test"]]

    return pd.concat(splits, ignore_index=True)


def clean_mage(df):
    # So we can look up a model's family by its name
    model_to_family = {}

    for family, models in families.items():
        for model in models:
            model_to_family[model] = family

    # MAGE uses 1 for human, we use 0. The domain is whatever comes before "_human" or "_machine_",
    # and only continuation-prompt texts get a model name
    df = df.assign(
        label=1 - df["label"],
        domain=df["src"].str.split("_human").str[0].str.split("_machine_").str[0],
        model=df["src"].str.split("_machine_continuation_").str[1],
    )

    # Keep the ten main domains, and only human texts or continuation-prompt machine texts
    is_human = df["src"].str.endswith("_human")
    df = df[df["domain"].isin(main_domains) & (is_human | df["model"].notna())]

    # Add each text's family, cut off the CMV footnote, then drop exact duplicates
    df = df.assign(
        family=df["model"].map(model_to_family),
        text=df["text"].str.replace(r"(?s)\s*Hello, [^.!]*[.!] This is a footnote from.*?(?:Happy CMVing!|$)\s*",
                                    " ", regex=True).str.strip(),
    ).drop_duplicates(subset="text")

    # Same normalisation, de-duplication and 10-word minimum as HC3
    df = preprocess(df)

    return df.assign(n_words=df["text"].str.split().str.len(), text_id=df.index)