import nltk
import pandas as pd
from hdbscan import HDBSCAN
from datetime import datetime
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import traceback
from tqdm import tqdm
import re

BERTOPIC_MODEL_BASE_PATH = "models/bertopic_v1"
ARTICLES_PATH = "data/articles/articles_body/articles.parquet"
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 5, 2)

def load_articles() -> pd.DataFrame:
    try:
        articles = pd.read_parquet(ARTICLES_PATH)
        if articles.empty:
            raise ValueError(f"No articles found in {ARTICLES_PATH}. Please check the path and content.")
        if 'url' in articles.columns:
            articles = articles.drop(columns=["url"])
        return articles
    except Exception as e:
        raise RuntimeError(f"Failed to load articles from {ARTICLES_PATH}: {e}")

def get_document_corpus(docs_df: pd.DataFrame) -> pd.Series:
    return docs_df["title"].apply(lambda x: re.sub(r"[^\w\s]", "", str(x)))

def get_stopwords() -> list[str]:
    try:
        nltk.data.find("corpora/stopwords")
    except nltk.downloader.DownloadError:
        nltk.download("stopwords")
    return nltk.corpus.stopwords.words("portuguese") + ["g1", "explica", "veja", "vídeo"]

class TopicModelManager:
    def __init__(self):
        self.stopwords = get_stopwords()

    def get_new_bertopic_model(self) -> BERTopic:
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        return BERTopic(
            language='multilingual',
            min_topic_size=10,
            nr_topics='auto',
            vectorizer_model=CountVectorizer(stop_words=self.stopwords, ngram_range=(1,1)),
            hdbscan_model=hdbscan_model,
        )

class SectionGenerator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.topic_model_manager = TopicModelManager()

    def process_day(self, current_date: datetime, articles_for_day: pd.DataFrame, topic_model: BERTopic) -> pd.DataFrame | None:
        day_str = current_date.strftime("%Y-%m-%d")

        if articles_for_day.empty:
            print(f"No articles found for {day_str}. Skipping.")
            return None

        if 'newsId' not in articles_for_day.columns or articles_for_day['newsId'].isnull().all():
            articles_for_day['newsId'] = [f"doc_{day_str}_{i}" for i in range(len(articles_for_day))]

        try:
            corpus = get_document_corpus(articles_for_day).tolist()
            topics, probs = topic_model.fit_transform(corpus)

            valid_topic_ids = [t for t in set(topics) if t != -1]
            num_topics = len(valid_topic_ids)

            if num_topics == 0:
                print(f"No meaningful topics found for {day_str} after processing.")
                return None

            df_res = pd.DataFrame({
                "timestamp": datetime.now(),
                "processing_date": day_str,
                "newsId": articles_for_day["newsId"].reset_index(drop=True),
                "doc": corpus,
                "topic": [str(t) for t in topics],
                "topic_name_list": [
                    [word for word, _ in topic_model.get_topic(t)] if t != -1 else []
                    for t in topics
                ],
            })

        except Exception as e:
            print(f"Error processing articles for {day_str}: {e}")
            traceback.print_exc()
            print("Skipping this day due to error.")
            return None

        return df_res if not df_res.empty else None

    def run(self):
        all_articles = load_articles()
        if 'publishDate' not in all_articles.columns:
            raise ValueError("Required column 'publishDate' not found in articles DataFrame.")
        all_articles['date_col'] = pd.to_datetime(all_articles['publishDate'])

        df_all_sections = pd.DataFrame()
        current_date = self.start_date

        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        for current_date in tqdm(date_range, desc="Processing days"):
            topic_model = self.topic_model_manager.get_new_bertopic_model()
            day_df = all_articles[all_articles['date_col'].dt.date == current_date.date()].copy()
            daily = self.process_day(current_date, day_df, topic_model)
            if daily is not None and not daily.empty:
                df_all_sections = pd.concat([df_all_sections, daily], ignore_index=True)

        if not df_all_sections.empty:
            topics_filename = f"topics_{self.start_date:%Y%m%d}_to_{self.end_date:%Y%m%d}_bertopic_v1.parquet"
            df_all_sections.to_parquet(topics_filename, index=False)
            print(f"Saved topics to {topics_filename}")

        return df_all_sections


if __name__ == "__main__":
    generator = SectionGenerator(start_date=START_DATE, end_date=END_DATE)
    final_df = generator.run()
    if not final_df.empty:
        print("\n--- Example of Generated Sections ---")
        print(final_df.head())
        print(f"\nTotal sections generated: {len(final_df)}")
