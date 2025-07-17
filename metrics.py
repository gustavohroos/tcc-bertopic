import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
from itertools import chain

def compute_daily_metrics(day, df_day):
    if day == '2023-05-01':
        return {}
    topics = df_day['topic'].astype(int)
    valid_topics = topics[topics != -1].unique()
    num_topics = len(valid_topics)
    num_unassigned = int((topics == -1).sum())
    topic_word_lists = [
        df_day[df_day['topic'].astype(int) == t]['topic_name_list'].iloc[0].tolist()
        for t in valid_topics
    ]

    tokenized_docs = [doc.split() for doc in df_day['doc']]
    gensim_dict = Dictionary(tokenized_docs)

    if topic_word_lists:
        cm = CoherenceModel(
            topics=topic_word_lists,
            texts=tokenized_docs,
            dictionary=gensim_dict,
            coherence='c_npmi',
            processes=1
        )
        npmi_score = cm.get_coherence()
        N = len(topic_word_lists[0])

        all_words = list(chain.from_iterable(topic_word_lists))
        td_score = len(set(all_words)) / (N * num_topics)
    else:
        npmi_score, td_score = 0.0, 0.0

    return {
        'day': day,
        'num_topics': num_topics,
        'doc_count': len(df_day),
        'num_unassigned': num_unassigned,
        'npmi': npmi_score,
        'td': td_score
    }

if __name__ == "__main__":
    filenames = [
        "results/online/topics_20230101_to_20230502_bertopic_online.parquet",
        "results/v1/topics_20230101_to_20230502_bertopic_v1.parquet",
        "results/v2/topics_20230101_to_20230502_bertopic_v2.parquet"
    ]

    for filename in filenames:
        topics_df = pd.read_parquet(filename)

        results = []
        groups = list(topics_df.groupby('processing_date'))
        for day, df_day in tqdm(groups, desc=f"Dias processados ({filename})", total=len(groups)):
            results.append(compute_daily_metrics(day, df_day))

        daily_metrics = pd.DataFrame(results)
        output_csv = filename.replace(".parquet", "_daily_metrics.csv")
        daily_metrics.to_csv(output_csv, index=False)
        print(f"Saved metrics to {output_csv}")
        print(daily_metrics)
