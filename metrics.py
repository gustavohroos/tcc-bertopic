import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
from itertools import chain
import re
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return []
    words = text.split()
    cleaned = [re.sub(r"[^\w\s]", "", w).strip().lower() for w in words]
    return [w for w in cleaned if w]

def clean_topic_words(words):
    if isinstance(words, str):
        tokens = words.split()
    elif isinstance(words, (list, np.ndarray)):
        tokens = list(words)
    else:
        return []
    cleaned = [
        re.sub(r"[^\w\s]", "", w).strip().lower()
        for w in tokens
        if isinstance(w, str) and w.strip()
    ]
    return cleaned

def compute_run_metrics(timestamp, df_run):
    topics = df_run['section'].astype(int)
    valid_topics = topics[topics != -1].unique()
    num_topics = len(valid_topics)
    num_unassigned = int((topics == -1).sum())

    tokenized_docs = [clean_text(doc) for doc in df_run['doc']]
    tokenized_docs = [doc for doc in tokenized_docs if doc]
    if not tokenized_docs:
        return {
            'timestamp': timestamp,
            'num_topics': num_topics,
            'doc_count': len(df_run),
            'num_unassigned': num_unassigned,
            'npmi': 0.0,
            'cv': 0.0,
            'td': 0.0
        }

    gensim_dict = Dictionary(tokenized_docs)

    raw_topic_word_lists = [
        clean_topic_words(df_run[df_run['section'].astype(int) == t]['section_name_list'].iloc[0])
        for t in valid_topics
    ]
    topic_word_lists = [
        [w for w in topic if w in gensim_dict.token2id]
        for topic in raw_topic_word_lists
    ]
    topic_word_lists = [topic for topic in topic_word_lists if len(topic) >= 2]

    if not topic_word_lists:
        print(f"[{timestamp}] Nenhum tópico válido após limpeza: {[len(t) for t in raw_topic_word_lists]}")
        return {
            'timestamp': timestamp,
            'num_topics': num_topics,
            'doc_count': len(df_run),
            'num_unassigned': num_unassigned,
            'npmi': 0.0,
            'cv': 0.0,
            'td': 0.0
        }

    cm_npmi = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized_docs,
        dictionary=gensim_dict,
        coherence='c_npmi',
        processes=1
    )
    npmi_score = cm_npmi.get_coherence()

    cm_cv = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized_docs,
        dictionary=gensim_dict,
        coherence='c_v',
        processes=1
    )
    cv_score = cm_cv.get_coherence()

    N = max(len(words) for words in topic_word_lists)
    all_words = list(chain.from_iterable(topic_word_lists))
    td_score = len(set(all_words)) / (N * len(topic_word_lists))

    return {
        'timestamp': timestamp,
        'num_topics': num_topics,
        'doc_count': len(df_run),
        'num_unassigned': num_unassigned,
        'npmi': npmi_score,
        'cv': cv_score,
        'td': td_score
    }

if __name__ == "__main__":
    filenames = [
        "results/online/topics_20230101_to_20230430_bertopic_online.csv",
        "results/v1/topics_20230101_to_20230430_bertopic_v1.csv",
        "results/v2/topics_20230101_to_20230430_bertopic_v2.csv",
        "results/ab/bertopic_v1_topics_globo_2025-07-19_2025-07-26.csv",
        "results/ab/bertopic_v2_topics_globo_2025-07-19_2025-07-26.csv",
        "results/ab/bertopic_v3_topics_globo_2025-07-19_2025-07-26.csv"
    ]

    for filename in filenames:
        df = pd.read_csv(filename)

        results = []
        groups = list(df.groupby('timestamp'))
        print(f"Processing {len(groups)} runs from {filename}...")

        for timestamp, df_run in tqdm(groups, desc=f"Runs processed ({filename})", total=len(groups)):
            results.append(compute_run_metrics(timestamp, df_run))

        metrics_df = pd.DataFrame(results)
        output_csv = filename.replace(".csv", "_run_metrics.csv")
        metrics_df.to_csv(output_csv, index=False)
        print(f"Saved metrics to {output_csv}")
        print(metrics_df)
